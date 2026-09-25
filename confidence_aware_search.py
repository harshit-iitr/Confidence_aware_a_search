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

    def _forward_mc_dropout(self, state_tensor_single: torch.Tensor) -> List[float]:
        """
        Executes M stochastic forward passes in a single batched parallel execution.
        """
        # Batch M copies of the single state
        state_batch = state_tensor_single.repeat(self.num_mc_samples, 1, 1, 1)  # (M, 5, H, W)
        inp = torch.cat([state_batch, self.goal_tensor_batch], dim=1)           # (M, 10, H, W)
        
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
        f2 = self.model.gap(att18).flatten(1)  # (M, 250)
        d2 = F.relu(self.model.fc1(f2))
        d2 = F.dropout(d2, p=self.dropout_p, training=True)
        
        out = self.model.fc2(d2).squeeze(-1)  # (M,)
        return out.cpu().tolist()

    def evaluate(self, state: np.ndarray) -> HeuristicEvaluation:
        # Check goal condition
        if SokobanEnv.is_goal(state, self.box_targets):
            return HeuristicEvaluation(h_blend=0.0, p_blend=0.0, mean_p=0.0, p_class=0.0, variance=0.0, lambda_conf=1.0)

        # 1. Classical Heuristic & Percentile
        h_class_raw = manhattan_distance_heuristic(state, self.box_targets)
        p_class = self.classical_tracker.get_percentile_and_insert(h_class_raw)
        
        # 2. Neural MC Dropout (M samples)
        s_np = author_state_to_tensor(state, self.box_targets, self.dim)
        with torch.no_grad():
            s_tensor = torch.from_numpy(s_np).unsqueeze(0).to(self.device)
            mc_outputs = self._forward_mc_dropout(s_tensor)  # M raw values

        # 3. Percentiles across M trackers
        mc_percentiles = [
            self.learned_trackers[m].get_percentile_and_insert(mc_outputs[m])
            for m in range(self.num_mc_samples)
        ]
        
        # 4. Mean & Rank Variance (Eq. 3 & 4)
        mean_p = float(np.mean(mc_percentiles))
        variance = float(np.var(mc_percentiles))  # in [0, 0.25]
        
        # 5. Confidence Gating (Eq. 5 & 6)
        lambda_raw = max(0.0, 1.0 - 4.0 * variance)
        lambda_conf = self.lambda_min + (1.0 - self.lambda_min) * lambda_raw  # in [lambda_min, 1.0]
        
        # 6. Rank-Space Convex Blend (Eq. 7 & 8)
        p_blend = lambda_conf * mean_p + (1.0 - lambda_conf) * p_class
        h_blend = p_blend * self.C
        
        return HeuristicEvaluation(
            h_blend=h_blend,
            p_blend=p_blend,
            mean_p=mean_p,
            p_class=p_class,
            variance=variance,
            lambda_conf=lambda_conf
        )


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
    max_expansions: int = 10000,
    max_time: float = 10.0,
    dim: int = 10
) -> Tuple[bool, int, float, List[int], float, Dict]:
    """
    Confidence-Aware A* Search using Rank-Blended Heuristic.
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
        for n_state, act, cost in zip(next_states, act_nos, costs):
            n_key = SokobanEnv.state_to_key(n_state)
            new_g = current.g + cost
            
            if n_key in closed_dict:
                continue
                
            if n_key not in open_dict or new_g < open_dict[n_key]:
                open_dict[n_key] = new_g
                h_eval = hybrid_heuristic.evaluate(n_state)
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
