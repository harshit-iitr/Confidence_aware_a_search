import os
import sys
import time
import argparse
import csv
import numpy as np
import torch
import heapq
from typing import List, Tuple, Dict, Optional, Any

from sokoban_env import SokobanEnv
from torch_model import ChrestienHeuristicNet
from classical_heuristics import manhattan_distance_heuristic
from confidence_aware_search import (
    ConfidenceAwareMCHeuristic,
    AdaptiveSearchNode,
    run_confidence_aware_astar,
    run_confidence_aware_gbfs,
    run_fixed_hybrid_astar,
    run_fixed_hybrid_gbfs,
    get_device,
    author_state_to_tensor
)

# -------------------------------------------------------------------------
# Node Class for Deterministic Baseline Search
# -------------------------------------------------------------------------
class SearchNode:
    __slots__ = ('state', 'key', 'g', 'h', 'f', 'parent_key', 'action')
    def __init__(self, state: np.ndarray, g: float, h: float, f: float, parent_key=None, action=0):
        self.state = state
        self.key = SokobanEnv.state_to_key(state)
        self.g = g
        self.h = h
        self.f = f
        self.parent_key = parent_key
        self.action = action


def verify_solution_plan(init_state: np.ndarray, actions: List[int], box_targets: List[Tuple[int, int]], dim: int = 10) -> bool:
    """
    Independently verifies that the sequence of actions is strictly legal
    and reaches an authentic goal state by simulating each action step-by-step.
    """
    curr = init_state.copy()
    for act in actions:
        valid_nexts, valid_acts, _ = SokobanEnv.get_neighbors(curr, box_targets, dim)
        if act not in valid_acts:
            return False
        act_idx = valid_acts.index(act)
        curr = valid_nexts[act_idx]
    return SokobanEnv.is_goal(curr, box_targets)


# -------------------------------------------------------------------------
# Batched Paper Search (A* and GBFS) matching Chrestien et al. NeurIPS 2023
# -------------------------------------------------------------------------
def run_batched_paper_search(
    init_state: np.ndarray,
    box_targets: List[Tuple[int, int]],
    model: torch.nn.Module,
    goal_tensor: torch.Tensor,
    device: torch.device,
    alg: str = "astar",
    max_time: float = 600.0,
    max_expansions: int = 100000,
    dim: int = 10
) -> Tuple[bool, int, float, List[int], float]:
    """
    Batched neural forward evaluation matching Chrestien et al. (NeurIPS 2023).
    A* priority:   f = g + h_nn, secondary: h_nn, tertiary: counter
    GBFS priority: f = h_nn,     secondary: counter
    """
    start_time = time.time()
    
    init_tensor = torch.from_numpy(author_state_to_tensor(init_state, box_targets, dim)).unsqueeze(0).to(device)
    with torch.no_grad():
        init_h = model(init_tensor, goal_tensor).item()
        
    init_f = init_h if alg == "gbfs" else (0.0 + init_h)
    start_node = SearchNode(init_state, g=0.0, h=init_h, f=init_f)
    
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
            # Batch forward pass on GPU/XPU
            non_goal_indices = []
            h_vals = [0.0] * len(candidates)
            for idx, c in enumerate(candidates):
                if SokobanEnv.is_goal(c[0], box_targets):
                    h_vals[idx] = 0.0
                else:
                    non_goal_indices.append(idx)
                    
            if non_goal_indices:
                tensors = [author_state_to_tensor(candidates[i][0], box_targets, dim) for i in non_goal_indices]
                batch_t = torch.from_numpy(np.stack(tensors)).to(device)
                goals_t = goal_tensor.repeat(len(non_goal_indices), 1, 1, 1)
                with torch.no_grad():
                    preds = model(batch_t, goals_t).flatten().cpu().tolist()
                for i, pred in zip(non_goal_indices, preds):
                    h_vals[i] = pred
                    
            for (n_state, n_key, act, new_g), h_val in zip(candidates, h_vals):
                open_dict[n_key] = new_g
                new_f = h_val if alg == "gbfs" else (new_g + h_val)
                child = SearchNode(n_state, g=new_g, h=h_val, f=new_f, parent_key=current.key, action=act)
                counter += 1
                heapq.heappush(open_heap, (child.f, child.h, counter, child))
                
    return False, expansions, float("inf"), [], time.time() - start_time


# -------------------------------------------------------------------------
# Classical Admissible A* (Hungarian Manhattan Heuristic)
# -------------------------------------------------------------------------
def run_classical_astar(
    init_state: np.ndarray,
    box_targets: List[Tuple[int, int]],
    max_time: float = 600.0,
    max_expansions: int = 150000,
    dim: int = 10
) -> Tuple[bool, int, float, List[int], float]:
    """
    Classical Admissible A* using Hungarian Manhattan distance.
    """
    start_time = time.time()
    init_h = manhattan_distance_heuristic(init_state, box_targets)
    start_node = SearchNode(init_state, g=0.0, h=init_h, f=init_h)
    
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
        for n_state, act, cost in zip(next_states, act_nos, costs):
            n_key = SokobanEnv.state_to_key(n_state)
            new_g = current.g + cost
            if n_key in closed_dict and closed_dict[n_key].g <= new_g:
                continue
            if n_key in open_dict and open_dict[n_key] <= new_g:
                continue
                
            open_dict[n_key] = new_g
            h_val = manhattan_distance_heuristic(n_state, box_targets)
            child = SearchNode(n_state, g=new_g, h=h_val, f=new_g + h_val, parent_key=current.key, action=act)
            counter += 1
            heapq.heappush(open_heap, (child.f, child.h, counter, child))
            
    return False, expansions, float("inf"), [], time.time() - start_time


# -------------------------------------------------------------------------
# Main Rigorous Benchmark Runner
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Rigorous 600s Benchmark on Sokoban Test Set")
    parser.add_argument("--instances", type=int, default=200, help="Number of instances to evaluate (default: 200, matching paper Table 1)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index in test set (default: 0)")
    parser.add_argument("--timeout", type=float, default=600.0, help="Search timeout in seconds (default: 600.0s = 10 mins)")
    parser.add_argument("--max_exp", type=int, default=100000, help="Max expansions per search (default: 100000)")
    parser.add_argument("--checkpoint", type=str, default="finalSok3_pytorch.pt", help="Path to pre-trained PyTorch weights")
    parser.add_argument("--output_csv", type=str, default="rigorous_benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--report_md", type=str, default="rigorous_benchmark_report.md", help="Output Markdown report path")
    parser.add_argument("--log_file", type=str, default="rigorous_benchmark_run.log", help="Log file path")
    args = parser.parse_args()

    # Setup dual logging (stdout + file)
    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "w", encoding="utf-8", buffering=1)
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(args.log_file)

    print("==================================================================", flush=True)
    print("      RIGOROUS 600-SECOND SOKOBAN BENCHMARK (CHRESTIEN ET AL.)     ", flush=True)
    print("==================================================================", flush=True)
    print(f" Instances to Evaluate  : {args.instances} (start index: {args.start_idx})", flush=True)
    print(f" Search Timeout Limit   : {args.timeout:.1f} seconds (10 minutes)", flush=True)
    print(f" Expansion Ceiling      : {args.max_exp} expansions", flush=True)
    print(f" Pretrained Checkpoint  : {args.checkpoint}", flush=True)

    # 1. Device Setup
    device = get_device()
    dev_name = "Intel Arc GPU (XPU)" if device.type == "xpu" else ("CUDA GPU" if device.type == "cuda" else "CPU")
    print(f" Hardware Backend       : {device} [{dev_name}]", flush=True)

    # 2. Dataset Setup & Leakage Filtering
    train_file = os.path.join("Optimize-Planning-Heuristics-to-Rank", "sokoban", "train", "states10_3box.txt")
    test_file = os.path.join("Optimize-Planning-Heuristics-to-Rank", "sokoban", "test", "states10test.txt")
    
    train_raw = set()
    with open(train_file, "r") as f:
        for line in f:
            nums = tuple(int(x) for x in line.split())
            if nums:
                train_raw.add(nums)
                
    raw_test_states = SokobanEnv.load_dataset(test_file)
    clean_test_states = [s for s in raw_test_states if tuple(s.flatten().tolist()) not in train_raw]
    
    print(f" Total Test States in File : {len(raw_test_states)}", flush=True)
    print(f" Clean Disjoint States     : {len(clean_test_states)} (Filtered {len(raw_test_states) - len(clean_test_states)} training duplicates)", flush=True)
    
    test_states = clean_test_states[args.start_idx : args.start_idx + args.instances]
    n_instances = len(test_states)
    print(f" Active Test Subset        : {n_instances} clean instances [{args.start_idx} to {args.start_idx + n_instances - 1}]\n", flush=True)

    # 3. Load Pre-Trained Model
    model = ChrestienHeuristicNet(dim=10).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 4. Setup CSV Logger
    csv_file = open(args.output_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "map_id", "algorithm", "solved", "expansions", "path_cost", "search_time_sec",
        "mean_lambda", "mean_variance", "plan_length", "verified"
    ])
    csv_file.flush()

    # Algorithms to Evaluate (7 distinct search configurations)
    algorithms = [
        "Classical A*",
        "Paper Learned A*",
        "Paper Learned GBFS",
        "Fixed 50/50 Hybrid A*",
        "Fixed 50/50 Hybrid GBFS",
        "Confidence-Aware Hybrid A* (Ours)",
        "Confidence-Aware Hybrid GBFS (Ours)"
    ]

    summary_data = {
        alg: {
            "sol": 0, "exp": [], "cost": [], "time": [],
            "verified": 0, "lambdas": [], "vars": []
        }
        for alg in algorithms
    }

    # 5. Main Benchmark Loop
    total_start = time.time()
    for map_idx, state in enumerate(test_states, 1):
        actual_id = args.start_idx + map_idx
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)
        goal_t = torch.from_numpy(author_state_to_tensor(goal_state, box_targets, 10)).unsqueeze(0).to(device)

        print(f"------------------------------------------------------------------", flush=True)
        print(f"Evaluating Map {map_idx:03d} / {n_instances} (Global ID: {actual_id:04d}) ...", flush=True)

        # [1] Classical Admissible A*
        c_sol, c_exp, c_cost, c_plan, c_time = run_classical_astar(
            state, box_targets, max_time=args.timeout, max_expansions=args.max_exp, dim=10
        )
        c_ver = verify_solution_plan(state, c_plan, box_targets, 10) if c_sol else False
        if c_sol and c_ver:
            summary_data["Classical A*"]["sol"] += 1
            summary_data["Classical A*"]["verified"] += 1
            summary_data["Classical A*"]["exp"].append(c_exp)
            summary_data["Classical A*"]["cost"].append(c_cost)
            summary_data["Classical A*"]["time"].append(c_time)
        csv_writer.writerow([actual_id, "Classical A*", c_sol, c_exp, c_cost, f"{c_time:.4f}", "", "", len(c_plan), c_ver])
        print(f"  [1] Classical A*           : {'SOLVED ('+str(c_exp)+' exp, '+str(round(c_cost,1))+' cost)' if c_sol else 'TIMEOUT'} in {c_time*1000:6.1f}ms", flush=True)

        # [2] Paper Learned A* (Chrestien et al. NeurIPS 2023)
        pa_sol, pa_exp, pa_cost, pa_plan, pa_time = run_batched_paper_search(
            state, box_targets, model, goal_t, device, alg="astar", max_time=args.timeout, max_expansions=args.max_exp, dim=10
        )
        pa_ver = verify_solution_plan(state, pa_plan, box_targets, 10) if pa_sol else False
        if pa_sol and pa_ver:
            summary_data["Paper Learned A*"]["sol"] += 1
            summary_data["Paper Learned A*"]["verified"] += 1
            summary_data["Paper Learned A*"]["exp"].append(pa_exp)
            summary_data["Paper Learned A*"]["cost"].append(pa_cost)
            summary_data["Paper Learned A*"]["time"].append(pa_time)
        csv_writer.writerow([actual_id, "Paper Learned A*", pa_sol, pa_exp, pa_cost, f"{pa_time:.4f}", "", "", len(pa_plan), pa_ver])
        print(f"  [2] Paper Learned A*       : {'SOLVED ('+str(pa_exp)+' exp, '+str(round(pa_cost,1))+' cost)' if pa_sol else 'TIMEOUT'} in {pa_time*1000:6.1f}ms", flush=True)

        # [3] Paper Learned GBFS (Chrestien et al. NeurIPS 2023)
        pg_sol, pg_exp, pg_cost, pg_plan, pg_time = run_batched_paper_search(
            state, box_targets, model, goal_t, device, alg="gbfs", max_time=args.timeout, max_expansions=args.max_exp, dim=10
        )
        pg_ver = verify_solution_plan(state, pg_plan, box_targets, 10) if pg_sol else False
        if pg_sol and pg_ver:
            summary_data["Paper Learned GBFS"]["sol"] += 1
            summary_data["Paper Learned GBFS"]["verified"] += 1
            summary_data["Paper Learned GBFS"]["exp"].append(pg_exp)
            summary_data["Paper Learned GBFS"]["cost"].append(pg_cost)
            summary_data["Paper Learned GBFS"]["time"].append(pg_time)
        csv_writer.writerow([actual_id, "Paper Learned GBFS", pg_sol, pg_exp, pg_cost, f"{pg_time:.4f}", "", "", len(pg_plan), pg_ver])
        print(f"  [3] Paper Learned GBFS     : {'SOLVED ('+str(pg_exp)+' exp, '+str(round(pg_cost,1))+' cost)' if pg_sol else 'TIMEOUT'} in {pg_time*1000:6.1f}ms", flush=True)

        # Setup Hybrid Heuristic Instance
        hybrid_h = ConfidenceAwareMCHeuristic(
            model=model, goal_state=goal_state, box_targets=box_targets,
            device=device, num_mc_samples=5, dropout_p=0.10, lambda_min=0.20, scale_constant_C=50.0, dim=10
        )

        # [4] Fixed 50/50 Hybrid A* (Ablation)
        fa_sol, fa_exp, fa_cost, fa_plan, fa_time = run_fixed_hybrid_astar(
            state, box_targets, hybrid_h, fixed_lambda=0.50, max_time=args.timeout, max_expansions=args.max_exp, dim=10
        )
        fa_ver = verify_solution_plan(state, fa_plan, box_targets, 10) if fa_sol else False
        if fa_sol and fa_ver:
            summary_data["Fixed 50/50 Hybrid A*"]["sol"] += 1
            summary_data["Fixed 50/50 Hybrid A*"]["verified"] += 1
            summary_data["Fixed 50/50 Hybrid A*"]["exp"].append(fa_exp)
            summary_data["Fixed 50/50 Hybrid A*"]["cost"].append(fa_cost)
            summary_data["Fixed 50/50 Hybrid A*"]["time"].append(fa_time)
        csv_writer.writerow([actual_id, "Fixed 50/50 Hybrid A*", fa_sol, fa_exp, fa_cost, f"{fa_time:.4f}", "0.50", "0.00", len(fa_plan), fa_ver])
        print(f"  [4] Fixed 50/50 Hybrid A*  : {'SOLVED ('+str(fa_exp)+' exp, '+str(round(fa_cost,1))+' cost)' if fa_sol else 'TIMEOUT'} in {fa_time*1000:6.1f}ms", flush=True)

        # Reset trackers for clean run
        hybrid_h.classical_tracker.reset()
        for t in hybrid_h.learned_trackers: t.reset()

        # [5] Fixed 50/50 Hybrid GBFS (Ablation)
        fg_sol, fg_exp, fg_cost, fg_plan, fg_time = run_fixed_hybrid_gbfs(
            state, box_targets, hybrid_h, fixed_lambda=0.50, max_time=args.timeout, max_expansions=args.max_exp, dim=10
        )
        fg_ver = verify_solution_plan(state, fg_plan, box_targets, 10) if fg_sol else False
        if fg_sol and fg_ver:
            summary_data["Fixed 50/50 Hybrid GBFS"]["sol"] += 1
            summary_data["Fixed 50/50 Hybrid GBFS"]["verified"] += 1
            summary_data["Fixed 50/50 Hybrid GBFS"]["exp"].append(fg_exp)
            summary_data["Fixed 50/50 Hybrid GBFS"]["cost"].append(fg_cost)
            summary_data["Fixed 50/50 Hybrid GBFS"]["time"].append(fg_time)
        csv_writer.writerow([actual_id, "Fixed 50/50 Hybrid GBFS", fg_sol, fg_exp, fg_cost, f"{fg_time:.4f}", "0.50", "0.00", len(fg_plan), fg_ver])
        print(f"  [5] Fixed 50/50 Hybrid GBFS: {'SOLVED ('+str(fg_exp)+' exp, '+str(round(fg_cost,1))+' cost)' if fg_sol else 'TIMEOUT'} in {fg_time*1000:6.1f}ms", flush=True)

        # Reset trackers for our dynamic method
        hybrid_h.classical_tracker.reset()
        for t in hybrid_h.learned_trackers: t.reset()

        # [6] Confidence-Aware Hybrid A* (Ours)
        ca_sol, ca_exp, ca_cost, ca_plan, ca_time, ca_stats = run_confidence_aware_astar(
            state, box_targets, hybrid_h, max_expansions=args.max_exp, max_time=args.timeout, dim=10
        )
        ca_ver = verify_solution_plan(state, ca_plan, box_targets, 10) if ca_sol else False
        if ca_sol and ca_ver:
            summary_data["Confidence-Aware Hybrid A* (Ours)"]["sol"] += 1
            summary_data["Confidence-Aware Hybrid A* (Ours)"]["verified"] += 1
            summary_data["Confidence-Aware Hybrid A* (Ours)"]["exp"].append(ca_exp)
            summary_data["Confidence-Aware Hybrid A* (Ours)"]["cost"].append(ca_cost)
            summary_data["Confidence-Aware Hybrid A* (Ours)"]["time"].append(ca_time)
            summary_data["Confidence-Aware Hybrid A* (Ours)"]["lambdas"].append(ca_stats["mean_lambda"])
            summary_data["Confidence-Aware Hybrid A* (Ours)"]["vars"].append(ca_stats["mean_variance"])
        csv_writer.writerow([
            actual_id, "Confidence-Aware Hybrid A* (Ours)", ca_sol, ca_exp, ca_cost, f"{ca_time:.4f}",
            f"{ca_stats['mean_lambda']:.4f}", f"{ca_stats['mean_variance']:.4f}", len(ca_plan), ca_ver
        ])
        ca_lbl = f"lambda={ca_stats['mean_lambda']:.2f}" if ca_sol else "timed out"
        print(f"  [6] Confidence-Aware A*    : {'SOLVED ('+str(ca_exp)+' exp, '+str(round(ca_cost,1))+' cost, '+ca_lbl+')' if ca_sol else 'TIMEOUT'} in {ca_time*1000:6.1f}ms", flush=True)

        # Reset trackers for our dynamic GBFS method
        hybrid_h.classical_tracker.reset()
        for t in hybrid_h.learned_trackers: t.reset()

        # [7] Confidence-Aware Hybrid GBFS (Ours)
        cg_sol, cg_exp, cg_cost, cg_plan, cg_time, cg_stats = run_confidence_aware_gbfs(
            state, box_targets, hybrid_h, max_expansions=args.max_exp, max_time=args.timeout, dim=10
        )
        cg_ver = verify_solution_plan(state, cg_plan, box_targets, 10) if cg_sol else False
        if cg_sol and cg_ver:
            summary_data["Confidence-Aware Hybrid GBFS (Ours)"]["sol"] += 1
            summary_data["Confidence-Aware Hybrid GBFS (Ours)"]["verified"] += 1
            summary_data["Confidence-Aware Hybrid GBFS (Ours)"]["exp"].append(cg_exp)
            summary_data["Confidence-Aware Hybrid GBFS (Ours)"]["cost"].append(cg_cost)
            summary_data["Confidence-Aware Hybrid GBFS (Ours)"]["time"].append(cg_time)
            summary_data["Confidence-Aware Hybrid GBFS (Ours)"]["lambdas"].append(cg_stats["mean_lambda"])
            summary_data["Confidence-Aware Hybrid GBFS (Ours)"]["vars"].append(cg_stats["mean_variance"])
        csv_writer.writerow([
            actual_id, "Confidence-Aware Hybrid GBFS (Ours)", cg_sol, cg_exp, cg_cost, f"{cg_time:.4f}",
            f"{cg_stats['mean_lambda']:.4f}", f"{cg_stats['mean_variance']:.4f}", len(cg_plan), cg_ver
        ])
        csv_file.flush()
        cg_lbl = f"lambda={cg_stats['mean_lambda']:.2f}" if cg_sol else "timed out"
        print(f"  [7] Confidence-Aware GBFS  : {'SOLVED ('+str(cg_exp)+' exp, '+str(round(cg_cost,1))+' cost, '+cg_lbl+')' if cg_sol else 'TIMEOUT'} in {cg_time*1000:6.1f}ms", flush=True)

    csv_file.close()
    total_elapsed = time.time() - total_start

    # -------------------------------------------------------------------------
    # Final Aggregate Reporting & Markdown Table
    # -------------------------------------------------------------------------
    print("\n==================================================================", flush=True)
    print("               RIGOROUS BENCHMARK FINAL SUMMARY TABLE              ", flush=True)
    print(f" Total Evaluation Duration: {total_elapsed:.1f}s ({total_elapsed/60:.2f} mins)", flush=True)
    print("==================================================================\n", flush=True)

    report_lines = [
        "# Rigorous 600-Second Sokoban Benchmark Report\n",
        f"**Hardware Platform:** {device} [{dev_name}]  ",
        f"**Benchmark Test Set:** {n_instances} instances (disjoint from training set)  ",
        f"**Search Timeout:** {args.timeout:.1f}s (10.0 minutes) | **Max Expansions:** {args.max_exp}  ",
        f"**Total Run Duration:** {total_elapsed/60:.2f} minutes\n",
        "## Comprehensive Performance Table\n",
        "| Algorithm | Solve Rate (%) | Verified (%) | Mean Expansions | Median Expansions | Mean Path Cost | Mean Time (ms) | Median Time (ms) | Mean Lambda |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for alg in algorithms:
        d = summary_data[alg]
        rate = (d["sol"] / n_instances) * 100
        ver_rate = (d["verified"] / n_instances) * 100
        mean_e = np.mean(d["exp"]) if d["exp"] else 0.0
        med_e = np.median(d["exp"]) if d["exp"] else 0.0
        mean_c = np.mean(d["cost"]) if d["cost"] else 0.0
        mean_t = np.mean(d["time"]) * 1000 if d["time"] else 0.0
        med_t = np.median(d["time"]) * 1000 if d["time"] else 0.0
        m_lambda = f"{np.mean(d['lambdas']):.3f}" if d["lambdas"] else "N/A"

        line = f"| **{alg}** | {rate:5.1f}% | {ver_rate:5.1f}% | {mean_e:8.1f} | {med_e:6.1f} | {mean_c:6.1f} | {mean_t:7.1f} | {med_t:7.1f} | {m_lambda} |"
        report_lines.append(line)
        print(line, flush=True)

    with open(args.report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nSaved results to '{args.output_csv}'", flush=True)
    print(f"Saved markdown report to '{args.report_md}'", flush=True)
    print("Benchmark complete!\n", flush=True)


if __name__ == "__main__":
    main()
