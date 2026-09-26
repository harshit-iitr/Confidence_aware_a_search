import os
import sys
import time
import bisect
import heapq
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, NamedTuple

from src.torch_model import ChrestienHeuristicNet
from src.sokoban_env import SokobanEnv
from src.classical_heuristics import manhattan_distance_heuristic
from src.confidence_aware_search import (
    PercentileTracker,
    HeuristicEvaluation,
    AdaptiveSearchNode,
    author_state_to_tensor,
    get_device
)


class DeepEnsembleHeuristic:
    """
    Confidence-Aware Rank-Blended Heuristic using a Deep Ensemble of M trained networks.
    Estimates true multi-basin epistemic uncertainty via cross-model rank variance.
    """
    def __init__(
        self,
        models: List[torch.nn.Module],
        goal_state: np.ndarray,
        box_targets: List[Tuple[int, int]],
        device: torch.device,
        lambda_min: float = 0.20,
        scale_constant_C: float = 50.0,
        dim: int = 10
    ):
        self.models = models
        self.M = len(models)
        self.device = device
        self.box_targets = box_targets
        self.lambda_min = lambda_min
        self.C = scale_constant_C
        self.dim = dim
        
        # M percentile trackers for ensemble members + 1 for classical heuristic
        self.ensemble_trackers = [PercentileTracker() for _ in range(self.M)]
        self.classical_tracker = PercentileTracker()
        
        # Goal tensor
        goal_np = author_state_to_tensor(goal_state, box_targets, dim)
        self.goal_tensor_single = torch.from_numpy(goal_np).unsqueeze(0).to(device)

    def evaluate_batch(self, states: List[np.ndarray], fixed_lambda: Optional[float] = None) -> List[HeuristicEvaluation]:
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
            
        K = len(active_states)
        tensors = [author_state_to_tensor(s, self.box_targets, self.dim) for s in active_states]
        batch_np = np.stack(tensors)
        batch_tensor = torch.from_numpy(batch_np).to(self.device)  # (K, 5, H, W)
        goal_batch = self.goal_tensor_single.repeat(K, 1, 1, 1)    # (K, 5, H, W)
        
        # Forward pass across all M models
        # ensemble_preds: shape (M, K)
        ensemble_preds = []
        with torch.no_grad():
            for m_idx in range(self.M):
                pred_m = self.models[m_idx](batch_tensor, goal_batch).cpu().tolist()
                if isinstance(pred_m, float):
                    pred_m = [pred_m]
                ensemble_preds.append(pred_m)
                
        # For each candidate state, compute ensemble variance and gating weight
        for i, (orig_idx, p_class) in enumerate(zip(active_indices, active_p_class)):
            m_vals = [ensemble_preds[m][i] for m in range(self.M)]
            m_percentiles = [
                self.ensemble_trackers[m].get_percentile_and_insert(m_vals[m])
                for m in range(self.M)
            ]
            mean_p = float(np.mean(m_percentiles))
            variance = float(np.var(m_percentiles))
            
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


def run_ensemble_astar(
    init_state: np.ndarray,
    box_targets: List[Tuple[int, int]],
    ensemble_heuristic: DeepEnsembleHeuristic,
    max_expansions: int = 15000,
    max_time: float = 600.0,
    dim: int = 10,
    fixed_lambda: Optional[float] = None
) -> Tuple[bool, int, float, List[int], float, Dict]:
    start_time = time.time()
    init_eval = ensemble_heuristic.evaluate(init_state, fixed_lambda=fixed_lambda)
    start_node = AdaptiveSearchNode(
        init_state, g=0.0, h=init_eval.h_blend, f=init_eval.h_blend,
        lambda_conf=init_eval.lambda_conf, variance=init_eval.variance
    )
    
    open_heap = []
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
            evals = ensemble_heuristic.evaluate_batch([c[0] for c in candidates], fixed_lambda=fixed_lambda)
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


def run_ensemble_gbfs(
    init_state: np.ndarray,
    box_targets: List[Tuple[int, int]],
    ensemble_heuristic: DeepEnsembleHeuristic,
    max_expansions: int = 15000,
    max_time: float = 600.0,
    dim: int = 10,
    fixed_lambda: Optional[float] = None
) -> Tuple[bool, int, float, List[int], float, Dict]:
    start_time = time.time()
    init_eval = ensemble_heuristic.evaluate(init_state, fixed_lambda=fixed_lambda)
    start_node = AdaptiveSearchNode(
        init_state, g=0.0, h=init_eval.p_blend, f=init_eval.p_blend,
        lambda_conf=init_eval.lambda_conf, variance=init_eval.variance
    )
    
    open_heap = []
    heapq.heappush(open_heap, (start_node.p_blend if hasattr(start_node, 'p_blend') else start_node.f, 0, start_node))
    open_dict = {start_node.key: 0.0}
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
            
        _, _, current = heapq.heappop(open_heap)
        if current.key in closed_dict:
            continue
        closed_dict[current.key] = current
        expansions += 1
        lambda_history.append(current.lambda_conf)
        var_history.append(current.variance)
        
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
            evals = ensemble_heuristic.evaluate_batch([c[0] for c in candidates], fixed_lambda=fixed_lambda)
            for (n_state, n_key, act, new_g), h_eval in zip(candidates, evals):
                open_dict[n_key] = new_g
                child = AdaptiveSearchNode(
                    n_state, g=new_g, h=h_eval.p_blend, f=h_eval.p_blend,
                    lambda_conf=h_eval.lambda_conf, variance=h_eval.variance,
                    parent_key=current.key, action=act
                )
                counter += 1
                heapq.heappush(open_heap, (child.f, counter, child))

    elapsed = time.time() - start_time
    stats = {
        "mean_lambda": float(np.mean(lambda_history)) if lambda_history else 0.0,
        "mean_variance": float(np.mean(var_history)) if var_history else 0.0
    }
    return False, expansions, float("inf"), [], elapsed, stats
