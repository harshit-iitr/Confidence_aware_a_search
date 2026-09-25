import time
import bisect
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, NamedTuple

from torch_model import ChrestienHeuristicNet
from sokoban_env import SokobanEnv
from classical_heuristics import manhattan_distance_heuristic

def get_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def author_state_to_tensor(state: np.ndarray, box_targets: List[Tuple[int, int]], dim: int = 10) -> np.ndarray:
    """Exact tensor representation matching paper's training convention."""
    box_rows, box_cols = np.where(state == 4)
    pos = list(zip(box_rows.tolist(), box_cols.tolist()))
    
    tensor = np.zeros((5, dim, dim), dtype=np.float32)
    for c in range(5):
        tensor[c] = (state == c).astype(np.float32)
    for r, c in box_targets:
        tensor[2, r, c] = 1.0
    for r, c in pos:
        tensor[2, r, c] = 1.0
    return tensor


class PercentileTracker:
    """
    Maintains a dynamic empirical CDF for O(log V) rank lookup and insertion.
    Maps arbitrary raw heuristic scalars to uniform rank percentiles in [0, 1].
    """
    def __init__(self):
        self.values = []

    def get_percentile_and_insert(self, val: float) -> float:
        n = len(self.values)
        if n == 0:
            self.values.append(val)
            return 0.5  # Neutral percentile for first state
        
        # Rank lookup: O(log V)
        rank = bisect.bisect_left(self.values, val)
        percentile = float(rank) / float(n)
        
        # Sorted insertion: O(log V) + O(V) list shift in Python (negligible for V < 50,000)
        bisect.insort(self.values, val)
        return percentile

    def reset(self):
        self.values.clear()


class HeuristicEvaluation(NamedTuple):
    h_blend: float
    p_blend: float
    mean_p: float
    p_class: float
    variance: float
    lambda_conf: float


class ConfidenceAwareMCHeuristic:
    """
    Confidence-Aware Rank-Blended Heuristic using Test-Time Monte Carlo Dropout.
    """
    def __init__(
        self,
        model: ChrestienHeuristicNet,
        goal_state: np.ndarray,
        box_targets: List[Tuple[int, int]],
        device: torch.device,
        num_mc_samples: int = 5,
        dropout_p: float = 0.10,
        lambda_min: float = 0.20,
        scale_constant_C: float = 50.0,
        dim: int = 10
    ):
        self.model = model
        self.device = device
        self.box_targets = box_targets
        self.num_mc_samples = num_mc_samples
        self.dropout_p = dropout_p
        self.lambda_min = lambda_min
        self.C = scale_constant_C
        self.dim = dim
        
        # Percentile trackers: M trackers for MC samples + 1 for classical heuristic
        self.learned_trackers = [PercentileTracker() for _ in range(num_mc_samples)]
        self.classical_tracker = PercentileTracker()
        
        # Goal tensor (cached on device)
        goal_np = author_state_to_tensor(goal_state, box_targets, dim)
        self.goal_tensor_single = torch.from_numpy(goal_np).unsqueeze(0).to(device)  # (1, 5, H, W)
        self.goal_tensor_batch = self.goal_tensor_single.repeat(num_mc_samples, 1, 1, 1)  # (M, 5, H, W)

    def _forward_mc_dropout_batch(self, states_tensor: torch.Tensor) -> List[List[float]]:
        """
        Executes M stochastic forward passes for K candidate states in a single parallel tensor batch.
        states_tensor: shape (K, 5, H, W)
        Returns: list of K lists, each containing M float predictions.
        """
        K = states_tensor.shape[0]
        if K == 0:
            return []
            
        # Repeat each candidate state M times: (K*M, 5, H, W)
        state_batch = torch.repeat_interleave(states_tensor, self.num_mc_samples, dim=0)
        # Repeat single goal tensor K*M times: (K*M, 5, H, W)
        goal_batch = self.goal_tensor_single.repeat(K * self.num_mc_samples, 1, 1, 1)
        inp = torch.cat([state_batch, goal_batch], dim=1)  # (K*M, 10, H, W)
        
        # Conv backbone
        x = F.relu(self.model.conv1(inp))
        x = F.relu(self.model.conv2(torch.cat([x, inp], dim=1)))
        x = F.relu(self.model.conv3(torch.cat([x, inp], dim=1)))
        x = F.relu(self.model.conv4(torch.cat([x, inp], dim=1)))
        x = F.relu(self.model.conv5(torch.cat([x, inp], dim=1)))
        x = F.relu(self.model.conv6(torch.cat([x, inp], dim=1)))
        x = F.relu(self.model.conv7(torch.cat([x, inp], dim=1)))
        
        # Attention blocks with mild stochastic test-time dropout
        p = F.relu(self.model.conv8(torch.cat([x, inp], dim=1)))
        att15 = self.model.att1(p)
        att15 = torch.cat([att15, self.model.pos1(p), inp], dim=1)
        att15 = F.dropout2d(att15, p=self.dropout_p, training=True)
        
        q = F.relu(self.model.conv9(att15))
        att16 = self.model.att2(q)
        att16 = torch.cat([att16, self.model.pos2(q), inp], dim=1)
        att16 = F.dropout2d(att16, p=self.dropout_p, training=True)
        
        r = F.relu(self.model.conv10(att16))
        att17 = self.model.att3(r)
        att17 = torch.cat([att17, self.model.pos3(r), inp], dim=1)
        att17 = F.dropout2d(att17, p=self.dropout_p, training=True)
        
        s = F.relu(self.model.conv11(att17))
        att18 = self.model.att4(s)
        att18 = torch.cat([att18, self.model.pos4(s), inp], dim=1)
        
        # Dense Head with latent dropout
        f2 = self.model.gap(att18).flatten(1)  # (K*M, 250)
        d2 = F.relu(self.model.fc1(f2))
        d2 = F.dropout(d2, p=self.dropout_p, training=True)
        
        out = self.model.fc2(d2).view(K, self.num_mc_samples)  # (K, M)
        return out.cpu().tolist()

    def evaluate_batch(self, states: List[np.ndarray], fixed_lambda: Optional[float] = None) -> List[HeuristicEvaluation]:
        """
        Batched evaluation of candidate child states.
        Combines classical Manhattan percentile tracking and parallel GPU MC dropout.
        If fixed_lambda is provided, uses fixed_lambda instead of dynamic rank variance gating.
        """
        if not states:
            return []
            
        results = [None] * len(states)
        active_indices = []
        active_states = []
        active_p_class = []
        
        for idx, s in enumerate(states):
            if SokobanEnv.is_goal(s, self.box_targets):
                results[idx] = HeuristicEvaluation(h_blend=0.0, p_blend=0.0, mean_p=0.0, p_class=0.0, variance=0.0, lambda_conf=1.0)
            else:
                active_indices.append(idx)
                active_states.append(s)
                h_class_raw = manhattan_distance_heuristic(s, self.box_targets)
                p_class = self.classical_tracker.get_percentile_and_insert(h_class_raw)
                active_p_class.append(p_class)
                
        if not active_indices:
            return results
            
        # Parallel GPU tensor forward pass
        tensors = [author_state_to_tensor(s, self.box_targets, self.dim) for s in active_states]
        batch_np = np.stack(tensors)
        with torch.no_grad():
            batch_tensor = torch.from_numpy(batch_np).to(self.device)
            mc_outputs_grid = self._forward_mc_dropout_batch(batch_tensor)  # (K_active, M)
            
        for i, (orig_idx, p_class) in enumerate(zip(active_indices, active_p_class)):
            mc_vals = mc_outputs_grid[i]
            mc_percentiles = [
                self.learned_trackers[m].get_percentile_and_insert(mc_vals[m])
                for m in range(self.num_mc_samples)
            ]
            mean_p = float(np.mean(mc_percentiles))
            variance = float(np.var(mc_percentiles))
            
            if fixed_lambda is not None:
                lambda_conf = fixed_lambda
            else:
                lambda_raw = max(0.0, 1.0 - 4.0 * variance)
                lambda_conf = self.lambda_min + (1.0 - self.lambda_min) * lambda_raw
                
            p_blend = lambda_conf * mean_p + (1.0 - lambda_conf) * p_class
            h_blend = p_blend * self.C
            
            results[orig_idx] = HeuristicEvaluation(
                h_blend=h_blend,
                p_blend=p_blend,
                mean_p=mean_p,
                p_class=p_class,
                variance=variance,
                lambda_conf=lambda_conf
            )
            
        return results

    def evaluate(self, state: np.ndarray, fixed_lambda: Optional[float] = None) -> HeuristicEvaluation:
        return self.evaluate_batch([state], fixed_lambda=fixed_lambda)[0]


class AdaptiveSearchNode:
    __slots__ = ('state', 'key', 'g', 'h', 'f', 'parent_key', 'action', 'lambda_conf', 'variance')
    def __init__(self, state: np.ndarray, g: float, h: float, f: float, lambda_conf: float, variance: float, parent_key=None, action=0):
        self.state = state
        self.key = SokobanEnv.state_to_key(state)
        self.g = g
        self.h = h
        self.f = f
        self.lambda_conf = lambda_conf
        self.variance = variance
        self.parent_key = parent_key
        self.action = action


def run_confidence_aware_astar(
    init_state: np.ndarray,
    box_targets: List[Tuple[int, int]],
    hybrid_heuristic: ConfidenceAwareMCHeuristic,
    max_expansions: int = 15000,
    max_time: float = 600.0,
    dim: int = 10
) -> Tuple[bool, int, float, List[int], float, Dict]:
    """
    Confidence-Aware A* Search using Rank-Blended Heuristic with Neighbor Batching.
    """
    start_time = time.time()
    
    init_eval = hybrid_heuristic.evaluate(init_state)
    start_node = AdaptiveSearchNode(
        init_state, g=0.0, h=init_eval.h_blend, f=init_eval.h_blend,
        lambda_conf=init_eval.lambda_conf, variance=init_eval.variance
    )
    
    open_heap = []
    # Min-heap key: (f, h, counter, node)
    heapq.heappush(open_heap, (start_node.f, start_node.h, 0, start_node))
    
    open_dict = {start_node.key: start_node.g}
    closed_dict = {}
    
    counter = 0
    expansions = 0
    
    lambda_history = []
    var_history = []

    while open_heap:
        if (time.time() - start_time) > max_time or expansions >= max_expansions:
            elapsed = time.time() - start_time
            stats = {
                "mean_lambda": float(np.mean(lambda_history)) if lambda_history else 0.0,
                "mean_variance": float(np.mean(var_history)) if var_history else 0.0
            }
            return False, expansions, float("inf"), [], elapsed, stats
            
        _, _, _, current = heapq.heappop(open_heap)
        if current.key in closed_dict:
            continue
        closed_dict[current.key] = current
        expansions += 1
        
        lambda_history.append(current.lambda_conf)
        var_history.append(current.variance)
        
        # Goal test
        if SokobanEnv.is_goal(current.state, box_targets):
            elapsed = time.time() - start_time
            actions = []
            curr = current
            while curr.parent_key is not None:
                actions.append(curr.action)
                curr = closed_dict[curr.parent_key]
            actions.reverse()
            stats = {
                "mean_lambda": float(np.mean(lambda_history)),
                "mean_variance": float(np.mean(var_history)),
                "min_lambda": float(np.min(lambda_history)),
                "max_variance": float(np.max(var_history))
            }
            return True, expansions, current.g, actions, elapsed, stats

        next_states, act_nos, costs = SokobanEnv.get_neighbors(current.state, box_targets, dim)
        candidates = []
        for n_state, act, cost in zip(next_states, act_nos, costs):
            n_key = SokobanEnv.state_to_key(n_state)
            new_g = current.g + cost
            if n_key in closed_dict and closed_dict[n_key].g <= new_g:
                continue
            if n_key in open_dict and open_dict[n_key] <= new_g:
                continue
            candidates.append((n_state, n_key, act, new_g))
            
        if candidates:
            evals = hybrid_heuristic.evaluate_batch([c[0] for c in candidates])
            for (n_state, n_key, act, new_g), h_eval in zip(candidates, evals):
                open_dict[n_key] = new_g
                new_f = new_g + h_eval.h_blend
                child = AdaptiveSearchNode(
                    n_state, g=new_g, h=h_eval.h_blend, f=new_f,
                    lambda_conf=h_eval.lambda_conf, variance=h_eval.variance,
                    parent_key=current.key, action=act
                )
                counter += 1
                heapq.heappush(open_heap, (child.f, child.h, counter, child))

    elapsed = time.time() - start_time
    stats = {
        "mean_lambda": float(np.mean(lambda_history)) if lambda_history else 0.0,
        "mean_variance": float(np.mean(var_history)) if var_history else 0.0
    }
    return False, expansions, float("inf"), [], elapsed, stats


def run_confidence_aware_gbfs(
    init_state: np.ndarray,
    box_targets: List[Tuple[int, int]],
    hybrid_heuristic: ConfidenceAwareMCHeuristic,
    max_expansions: int = 15000,
    max_time: float = 600.0,
    dim: int = 10
) -> Tuple[bool, int, float, List[int], float, Dict]:
    """
    Confidence-Aware GBFS Search using Pure Rank-Blended Heuristic (No cost scaling constant needed).
    Priority: f(s) = p_blend(s) in [0, 1].
    Secondary tie-breaker: p_class(s).
    """
    start_time = time.time()
    
    init_eval = hybrid_heuristic.evaluate(init_state)
    start_node = AdaptiveSearchNode(
        init_state, g=0.0, h=init_eval.p_blend, f=init_eval.p_blend,
        lambda_conf=init_eval.lambda_conf, variance=init_eval.variance
    )
    
    open_heap = []
    # Min-heap key: (f=p_blend, secondary=p_class, counter, node)
    heapq.heappush(open_heap, (start_node.f, init_eval.p_class, 0, start_node))
    
    open_dict = {start_node.key: start_node.g}
    closed_dict = {}
    
    counter = 0
    expansions = 0
    
    lambda_history = []
    var_history = []

    while open_heap:
        if (time.time() - start_time) > max_time or expansions >= max_expansions:
            elapsed = time.time() - start_time
            stats = {
                "mean_lambda": float(np.mean(lambda_history)) if lambda_history else 0.0,
                "mean_variance": float(np.mean(var_history)) if var_history else 0.0
            }
            return False, expansions, float("inf"), [], elapsed, stats
            
        _, _, _, current = heapq.heappop(open_heap)
        if current.key in closed_dict:
            continue
        closed_dict[current.key] = current
        expansions += 1
        
        lambda_history.append(current.lambda_conf)
        var_history.append(current.variance)
        
        # Goal test
        if SokobanEnv.is_goal(current.state, box_targets):
            elapsed = time.time() - start_time
            actions = []
            curr = current
            while curr.parent_key is not None:
                actions.append(curr.action)
                curr = closed_dict[curr.parent_key]
            actions.reverse()
            stats = {
                "mean_lambda": float(np.mean(lambda_history)),
                "mean_variance": float(np.mean(var_history)),
                "min_lambda": float(np.min(lambda_history)),
                "max_variance": float(np.max(var_history))
            }
            return True, expansions, current.g, actions, elapsed, stats

        next_states, act_nos, costs = SokobanEnv.get_neighbors(current.state, box_targets, dim)
        candidates = []
        for n_state, act, cost in zip(next_states, act_nos, costs):
            n_key = SokobanEnv.state_to_key(n_state)
            new_g = current.g + cost
            if n_key in closed_dict and closed_dict[n_key].g <= new_g:
                continue
            if n_key in open_dict and open_dict[n_key] <= new_g:
                continue
            candidates.append((n_state, n_key, act, new_g))
            
        if candidates:
            evals = hybrid_heuristic.evaluate_batch([c[0] for c in candidates])
            for (n_state, n_key, act, new_g), h_eval in zip(candidates, evals):
                open_dict[n_key] = new_g
                new_f = h_eval.p_blend  # Pure rank blend in [0, 1]
                child = AdaptiveSearchNode(
                    n_state, g=new_g, h=h_eval.p_blend, f=new_f,
                    lambda_conf=h_eval.lambda_conf, variance=h_eval.variance,
                    parent_key=current.key, action=act
                )
                counter += 1
                heapq.heappush(open_heap, (child.f, h_eval.p_class, counter, child))

    elapsed = time.time() - start_time
    stats = {
        "mean_lambda": float(np.mean(lambda_history)) if lambda_history else 0.0,
        "mean_variance": float(np.mean(var_history)) if var_history else 0.0
    }
    return False, expansions, float("inf"), [], elapsed, stats


def run_fixed_hybrid_astar(
    init_state: np.ndarray,
    box_targets: List[Tuple[int, int]],
    hybrid_heuristic: ConfidenceAwareMCHeuristic,
    fixed_lambda: float = 0.50,
    max_expansions: int = 15000,
    max_time: float = 600.0,
    dim: int = 10
) -> Tuple[bool, int, float, List[int], float]:
    """
    Fixed-Weight Hybrid A* Ablation with Batched Candidate Evaluation (Static lambda, e.g. 0.50).
    """
    start_time = time.time()
    init_eval = hybrid_heuristic.evaluate(init_state, fixed_lambda=fixed_lambda)
    start_node = AdaptiveSearchNode(
        init_state, g=0.0, h=init_eval.h_blend, f=init_eval.h_blend,
        lambda_conf=fixed_lambda, variance=0.0
    )
    
    open_heap = []
    heapq.heappush(open_heap, (start_node.f, start_node.h, 0, start_node))
    open_dict = {start_node.key: start_node.g}
    closed_dict = {}
    
    counter = 0
    expansions = 0
    
    while open_heap:
        if (time.time() - start_time) > max_time or expansions >= max_expansions:
            return False, expansions, float("inf"), [], time.time() - start_time
            
        _, _, _, current = heapq.heappop(open_heap)
        if current.key in closed_dict:
            continue
        closed_dict[current.key] = current
        expansions += 1
        
        if SokobanEnv.is_goal(current.state, box_targets):
            elapsed = time.time() - start_time
            actions = []
            curr = current
            while curr.parent_key is not None:
                actions.append(curr.action)
                curr = closed_dict[curr.parent_key]
            actions.reverse()
            return True, expansions, current.g, actions, elapsed

        next_states, act_nos, costs = SokobanEnv.get_neighbors(current.state, box_targets, dim)
        candidates = []
        for n_state, act, cost in zip(next_states, act_nos, costs):
            n_key = SokobanEnv.state_to_key(n_state)
            new_g = current.g + cost
            if n_key in closed_dict and closed_dict[n_key].g <= new_g:
                continue
            if n_key in open_dict and open_dict[n_key] <= new_g:
                continue
            candidates.append((n_state, n_key, act, new_g))
            
        if candidates:
            evals = hybrid_heuristic.evaluate_batch([c[0] for c in candidates], fixed_lambda=fixed_lambda)
            for (n_state, n_key, act, new_g), h_eval in zip(candidates, evals):
                open_dict[n_key] = new_g
                new_f = new_g + h_eval.h_blend
                child = AdaptiveSearchNode(
                    n_state, g=new_g, h=h_eval.h_blend, f=new_f,
                    lambda_conf=fixed_lambda, variance=0.0,
                    parent_key=current.key, action=act
                )
                counter += 1
                heapq.heappush(open_heap, (child.f, child.h, counter, child))

    return False, expansions, float("inf"), [], time.time() - start_time


def run_fixed_hybrid_gbfs(
    init_state: np.ndarray,
    box_targets: List[Tuple[int, int]],
    hybrid_heuristic: ConfidenceAwareMCHeuristic,
    fixed_lambda: float = 0.50,
    max_expansions: int = 15000,
    max_time: float = 600.0,
    dim: int = 10
) -> Tuple[bool, int, float, List[int], float]:
    """
    Fixed-Weight Hybrid GBFS Ablation with Batched Candidate Evaluation (Static lambda, e.g. 0.50).
    """
    start_time = time.time()
    init_eval = hybrid_heuristic.evaluate(init_state, fixed_lambda=fixed_lambda)
    start_node = AdaptiveSearchNode(
        init_state, g=0.0, h=init_eval.p_blend, f=init_eval.p_blend,
        lambda_conf=fixed_lambda, variance=0.0
    )
    
    open_heap = []
    heapq.heappush(open_heap, (start_node.f, init_eval.p_class, 0, start_node))
    open_dict = {start_node.key: start_node.g}
    closed_dict = {}
    
    counter = 0
    expansions = 0
    
    while open_heap:
        if (time.time() - start_time) > max_time or expansions >= max_expansions:
            return False, expansions, float("inf"), [], time.time() - start_time
            
        _, _, _, current = heapq.heappop(open_heap)
        if current.key in closed_dict:
            continue
        closed_dict[current.key] = current
        expansions += 1
        
        if SokobanEnv.is_goal(current.state, box_targets):
            elapsed = time.time() - start_time
            actions = []
            curr = current
            while curr.parent_key is not None:
                actions.append(curr.action)
                curr = closed_dict[curr.parent_key]
            actions.reverse()
            return True, expansions, current.g, actions, elapsed

        next_states, act_nos, costs = SokobanEnv.get_neighbors(current.state, box_targets, dim)
        candidates = []
        for n_state, act, cost in zip(next_states, act_nos, costs):
            n_key = SokobanEnv.state_to_key(n_state)
            new_g = current.g + cost
            if n_key in closed_dict and closed_dict[n_key].g <= new_g:
                continue
            if n_key in open_dict and open_dict[n_key] <= new_g:
                continue
            candidates.append((n_state, n_key, act, new_g))
            
        if candidates:
            evals = hybrid_heuristic.evaluate_batch([c[0] for c in candidates], fixed_lambda=fixed_lambda)
            for (n_state, n_key, act, new_g), h_eval in zip(candidates, evals):
                open_dict[n_key] = new_g
                new_f = h_eval.p_blend
                child = AdaptiveSearchNode(
                    n_state, g=new_g, h=h_eval.p_blend, f=new_f,
                    lambda_conf=fixed_lambda, variance=0.0,
                    parent_key=current.key, action=act
                )
                counter += 1
                heapq.heappush(open_heap, (child.f, h_eval.p_class, counter, child))

    return False, expansions, float("inf"), [], time.time() - start_time

