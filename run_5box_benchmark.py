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
            cand_tensors = np.stack([author_state_to_tensor(c[0], box_targets, dim) for c in candidates])
            cand_t = torch.from_numpy(cand_tensors).to(device)
            B = cand_t.shape[0]
            g_batch = goal_tensor.repeat(B, 1, 1, 1)
            with torch.no_grad():
                h_vals = model(cand_t, g_batch).cpu().tolist()
            if isinstance(h_vals, float):
                h_vals = [h_vals]
                
            for (n_state, n_key, act, new_g), h_val in zip(candidates, h_vals):
                open_dict[n_key] = new_g
                f_val = h_val if alg == "gbfs" else (new_g + h_val)
                child = SearchNode(n_state, g=new_g, h=h_val, f=f_val, parent_key=current.key, action=act)
                counter += 1
                heapq.heappush(open_heap, (child.f, child.h, counter, child))

    return False, expansions, float("inf"), [], time.time() - start_time


# -------------------------------------------------------------------------
# Classical Admissible A* Search (Manhattan Distance)
# -------------------------------------------------------------------------
def run_classical_astar(
    init_state: np.ndarray,
    box_targets: List[Tuple[int, int]],
    max_time: float = 600.0,
    max_expansions: int = 150000,
    dim: int = 10
) -> Tuple[bool, int, float, List[int], float]:
    """
    Standard Classical A* with Manhattan distance heuristic.
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
            
            h_val = manhattan_distance_heuristic(n_state, box_targets)
            f_val = new_g + h_val
            open_dict[n_key] = new_g
            child = SearchNode(n_state, g=new_g, h=h_val, f=f_val, parent_key=current.key, action=act)
            counter += 1
            heapq.heappush(open_heap, (child.f, child.h, counter, child))

    return False, expansions, float("inf"), [], time.time() - start_time


# -------------------------------------------------------------------------
# Interim & Final Statistics Printer
# -------------------------------------------------------------------------
def print_interim_statistics(summary_data: Dict[str, Dict], map_idx: int, total_maps: int, start_time: float, is_final: bool = False):
    """
    Prints aggregated progress and statistical metrics to stdout after every N runs.
    Uses strict plain ASCII characters for safe Windows console output.
    """
    elapsed = time.time() - start_time
    tag = "FINAL BENCHMARK REPORT" if is_final else f"INTERIM CHECKPOINT"
    print("\n" + "=" * 88, flush=True)
    print(f"  >>> {tag} AFTER {map_idx} / {total_maps} RUNS (Elapsed: {elapsed/60:.2f} min) <<<", flush=True)
    print("=" * 88, flush=True)
    print(f"{'Algorithm':<33} | {'Solved':<10} | {'Exp (Mean)':<11} | {'Exp (Med)':<10} | {'Cost':<7} | {'Time':<9} | {'Verified'}", flush=True)
    print("-" * 88, flush=True)
    
    for alg, d in summary_data.items():
        sol_count = d["sol"]
        sol_pct = (sol_count / map_idx) * 100.0 if map_idx > 0 else 0.0
        sol_str = f"{sol_count}/{map_idx} ({sol_pct:.0f}%)"
        if sol_count > 0:
            mean_exp = f"{np.mean(d['exp']):.1f}"
            med_exp = f"{np.median(d['exp']):.1f}"
            mean_cost = f"{np.mean(d['cost']):.1f}"
            mean_time = f"{np.mean(d['time']):.3f}s"
            ver_pct = f"{(d['verified']/sol_count)*100:.0f}%"
        else:
            mean_exp, med_exp, mean_cost, mean_time, ver_pct = "N/A", "N/A", "N/A", "N/A", "N/A"
            
        print(f"{alg:<33} | {sol_str:<10} | {mean_exp:<11} | {med_exp:<10} | {mean_cost:<7} | {mean_time:<9} | {ver_pct}", flush=True)
    print("-" * 88, flush=True)
    
    # Head-to-Head: Confidence-Aware Hybrid A* vs Paper Learned A*
    if "Confidence-Aware Hybrid A* (Ours)" in summary_data and "Paper Learned A*" in summary_data:
        ours_exp = summary_data["Confidence-Aware Hybrid A* (Ours)"]["exp"]
        paper_exp = summary_data["Paper Learned A*"]["exp"]
        if len(ours_exp) == len(paper_exp) and len(ours_exp) > 0:
            wins = sum(1 for o, p in zip(ours_exp, paper_exp) if o < p)
            ties = sum(1 for o, p in zip(ours_exp, paper_exp) if o == p)
            losses = sum(1 for o, p in zip(ours_exp, paper_exp) if o > p)
            reduction = ((1.0 - (np.mean(ours_exp) / np.mean(paper_exp))) * 100.0) if np.mean(paper_exp) > 0 else 0.0
            med_reduction = ((1.0 - (np.median(ours_exp) / np.median(paper_exp))) * 100.0) if np.median(paper_exp) > 0 else 0.0
            
            p_val_str = "N/A"
            try:
                from scipy.stats import wilcoxon
                diffs = [p - o for p, o in zip(paper_exp, ours_exp)]
                if any(d != 0 for d in diffs):
                    _, p_val = wilcoxon(paper_exp, ours_exp)
                    p_val_str = f"{p_val:.2e}"
            except Exception:
                pass
                
            print(f"  [HEAD-TO-HEAD: Confidence-Aware A* (Ours) vs Paper Learned A* on {len(ours_exp)} instances]", flush=True)
            print(f"    - Win Rate              : {wins} Wins / {ties} Ties / {losses} Losses ({wins/len(ours_exp)*100:.1f}% win rate)", flush=True)
            print(f"    - Mean Expansions       : {np.mean(ours_exp):.1f} vs {np.mean(paper_exp):.1f} ({reduction:+.2f}% node reduction)", flush=True)
            print(f"    - Median Expansions     : {np.median(ours_exp):.1f} vs {np.median(paper_exp):.1f} ({med_reduction:+.2f}% node reduction)", flush=True)
            print(f"    - Wilcoxon Signed-Rank  : p = {p_val_str}", flush=True)
            if summary_data["Confidence-Aware Hybrid A* (Ours)"]["lambdas"]:
                mean_lam = np.mean(summary_data["Confidence-Aware Hybrid A* (Ours)"]["lambdas"])
                mean_var = np.mean(summary_data["Confidence-Aware Hybrid A* (Ours)"]["vars"])
                print(f"    - Mean Gating Weight    : lambda = {mean_lam:.3f}, rank variance = {mean_var:.4f}", flush=True)

    # Head-to-Head: Confidence-Aware Hybrid GBFS vs Paper Learned GBFS
    if "Confidence-Aware Hybrid GBFS (Ours)" in summary_data and "Paper Learned GBFS" in summary_data:
        ours_g = summary_data["Confidence-Aware Hybrid GBFS (Ours)"]["exp"]
        paper_g = summary_data["Paper Learned GBFS"]["exp"]
        if len(ours_g) == len(paper_g) and len(ours_g) > 0:
            g_wins = sum(1 for o, p in zip(ours_g, paper_g) if o < p)
            g_ties = sum(1 for o, p in zip(ours_g, paper_g) if o == p)
            g_losses = sum(1 for o, p in zip(ours_g, paper_g) if o > p)
            g_red = ((1.0 - (np.mean(ours_g) / np.mean(paper_g))) * 100.0) if np.mean(paper_g) > 0 else 0.0
            print(f"  [HEAD-TO-HEAD: Confidence-Aware GBFS (Ours) vs Paper Learned GBFS on {len(ours_g)} instances]", flush=True)
            print(f"    - Win Rate              : {g_wins} Wins / {g_ties} Ties / {g_losses} Losses ({g_wins/len(ours_g)*100:.1f}% win rate)", flush=True)
            print(f"    - Mean Expansions       : {np.mean(ours_g):.1f} vs {np.mean(paper_g):.1f} ({g_red:+.2f}% node reduction)", flush=True)
    print("=" * 88 + "\n", flush=True)


# -------------------------------------------------------------------------
# Markdown Report Generator
# -------------------------------------------------------------------------
def generate_markdown_report(summary_data: Dict[str, Dict], n_total: int, total_time: float, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 5-Box Sokoban Out-of-Distribution (OOD) Benchmark Report\n\n")
        f.write("## 1. Experimental Overview\n")
        f.write(f"- **Benchmark Domain**: 10x10 Sokoban with 5 boxes (Out-of-Distribution zero-shot evaluation)\n")
        f.write(f"- **Dataset Size**: {n_total} procedurally generated mazes (`gym-sokoban` reverse-walk)\n")
        f.write(f"- **Total Benchmark Wall Time**: {total_time/60:.2f} minutes\n")
        f.write(f"- **Independent Solution Verification**: 100% physically verified via step-by-step transition replay\n")
        f.write(f"- **Pretrained Model**: Chrestien et al. `finalSok3` (trained strictly on 3 boxes, evaluated zero-shot)\n\n")
        
        f.write("## 2. Quantitative Performance Comparison\n\n")
        f.write("| Algorithm | Solve Rate | Expansions (Mean) | Expansions (Median) | Path Cost | Search Time (s) | Plan Verified |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        
        for alg, d in summary_data.items():
            sol = d["sol"]
            sol_pct = (sol / n_total) * 100.0 if n_total > 0 else 0.0
            if sol > 0:
                m_exp = f"{np.mean(d['exp']):.1f}"
                med_exp = f"{np.median(d['exp']):.1f}"
                m_cost = f"{np.mean(d['cost']):.1f}"
                m_time = f"{np.mean(d['time']):.3f}s"
                ver = f"{(d['verified']/sol)*100:.0f}%"
            else:
                m_exp, med_exp, m_cost, m_time, ver = "N/A", "N/A", "N/A", "N/A", "N/A"
            f.write(f"| **{alg}** | {sol}/{n_total} ({sol_pct:.1f}%) | {m_exp} | {med_exp} | {m_cost} | {m_time} | {ver} |\n")
        
        f.write("\n## 3. Head-to-Head Statistical Hypothesis Testing\n\n")
        if "Confidence-Aware Hybrid A* (Ours)" in summary_data and "Paper Learned A*" in summary_data:
            o_exp = summary_data["Confidence-Aware Hybrid A* (Ours)"]["exp"]
            p_exp = summary_data["Paper Learned A*"]["exp"]
            if len(o_exp) == len(p_exp) and len(o_exp) > 0:
                wins = sum(1 for o, p in zip(o_exp, p_exp) if o < p)
                ties = sum(1 for o, p in zip(o_exp, p_exp) if o == p)
                losses = sum(1 for o, p in zip(o_exp, p_exp) if o > p)
                red = ((1.0 - (np.mean(o_exp) / np.mean(p_exp))) * 100.0) if np.mean(p_exp) > 0 else 0.0
                med_red = ((1.0 - (np.median(o_exp) / np.median(p_exp))) * 100.0) if np.median(p_exp) > 0 else 0.0
                
                p_str = "N/A"
                try:
                    from scipy.stats import wilcoxon
                    _, p_val = wilcoxon(p_exp, o_exp)
                    p_str = f"{p_val:.2e}"
                except Exception:
                    pass
                    
                f.write("### Confidence-Aware Hybrid A* (Ours) vs Paper Learned A*\n")
                f.write(f"- **Head-to-Head Win Rate**: **{wins}/{len(o_exp)} ({wins/len(o_exp)*100:.1f}%)**\n")
                f.write(f"- **Mean Node Reduction**: **{red:+.2f}%** ({np.mean(o_exp):.1f} vs {np.mean(p_exp):.1f})\n")
                f.write(f"- **Median Node Reduction**: **{med_red:+.2f}%** ({np.median(o_exp):.1f} vs {np.median(p_exp):.1f})\n")
                f.write(f"- **Wilcoxon Signed-Rank Test**: p-value = **{p_str}**\n\n")


# -------------------------------------------------------------------------
# Main Execution Entrypoint
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="5-Box Sokoban OOD Benchmark")
    parser.add_argument("--dataset", type=str, default="sokoban_5box_test.txt", help="Path to 5-box dataset file")
    parser.add_argument("--model_path", type=str, default="finalSok3_pytorch.pt", help="Path to pretrained model")
    parser.add_argument("--num_mazes", type=int, default=100, help="Number of instances to evaluate")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-algorithm timeout in seconds")
    parser.add_argument("--max_exp", type=int, default=15000, help="Max node expansions for neural search")
    parser.add_argument("--classical_max_exp", type=int, default=150000, help="Max node expansions for classical A*")
    parser.add_argument("--log_interval", type=int, default=20, help="Print summary statistics every N runs")
    parser.add_argument("--output_csv", type=str, default="sokoban_5box_benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--output_report", type=str, default="sokoban_5box_benchmark_report.md", help="Output markdown path")
    args = parser.parse_args()

    print("=" * 88, flush=True)
    print("5-BOX SOKOBAN OUT-OF-DISTRIBUTION (OOD) BENCHMARK", flush=True)
    print(f"Dataset          : {args.dataset}", flush=True)
    print(f"Target Instances : {args.num_mazes}", flush=True)
    print(f"Timeout Limit    : {args.timeout} seconds", flush=True)
    print(f"Log Interval     : Every {args.log_interval} completed runs", flush=True)
    print(f"Neural Max Exp   : {args.max_exp}", flush=True)
    print(f"Classical Max Exp: {args.classical_max_exp}", flush=True)
    print("=" * 88, flush=True)

    device = get_device()
    print(f"Selected Compute Device: {device}", flush=True)

    if not os.path.exists(args.dataset):
        fallback = os.path.join("data", args.dataset)
        if os.path.exists(fallback):
            args.dataset = fallback
        else:
            raise FileNotFoundError(f"Dataset file '{args.dataset}' does not exist! Please run generate_5box_dataset.py first.")

    all_states = SokobanEnv.load_dataset(args.dataset)
    test_states = all_states[:args.num_mazes]
    n_instances = len(test_states)
    print(f"Loaded {n_instances} 5-box Sokoban instances for benchmarking.\n", flush=True)

    print(f"Loading pretrained model weights from '{args.model_path}'...", flush=True)
    model = ChrestienHeuristicNet(dim=10).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model successfully loaded and set to evaluation mode.\n", flush=True)

    csv_file = open(args.output_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "map_id", "algorithm", "solved", "expansions", "cost", "time_sec",
        "mean_lambda", "mean_variance", "plan_length", "verified"
    ])

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

    total_start = time.time()
    for map_idx, state in enumerate(test_states, 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)
        goal_t = torch.from_numpy(author_state_to_tensor(goal_state, box_targets, 10)).unsqueeze(0).to(device)

        print(f"------------------------------------------------------------------", flush=True)
        print(f"Evaluating Map {map_idx:03d} / {n_instances} ...", flush=True)

        # [1] Classical A*
        c_sol, c_exp, c_cost, c_plan, c_time = run_classical_astar(
            state, box_targets, max_time=args.timeout, max_expansions=args.classical_max_exp, dim=10
        )
        c_ver = verify_solution_plan(state, c_plan, box_targets, 10) if c_sol else False
        if c_sol and c_ver:
            summary_data["Classical A*"]["sol"] += 1
            summary_data["Classical A*"]["verified"] += 1
            summary_data["Classical A*"]["exp"].append(c_exp)
            summary_data["Classical A*"]["cost"].append(c_cost)
            summary_data["Classical A*"]["time"].append(c_time)
        csv_writer.writerow([map_idx, "Classical A*", c_sol, c_exp, c_cost, f"{c_time:.4f}", "", "", len(c_plan), c_ver])
        print(f"  [1] Classical A*           : {'SOLVED ('+str(c_exp)+' exp, '+str(round(c_cost,1))+' cost)' if c_sol else 'TIMEOUT'} in {c_time*1000:6.1f}ms", flush=True)

        # [2] Paper Learned A*
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
        csv_writer.writerow([map_idx, "Paper Learned A*", pa_sol, pa_exp, pa_cost, f"{pa_time:.4f}", "", "", len(pa_plan), pa_ver])
        print(f"  [2] Paper Learned A*       : {'SOLVED ('+str(pa_exp)+' exp, '+str(round(pa_cost,1))+' cost)' if pa_sol else 'TIMEOUT'} in {pa_time*1000:6.1f}ms", flush=True)

        # [3] Paper Learned GBFS
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
        csv_writer.writerow([map_idx, "Paper Learned GBFS", pg_sol, pg_exp, pg_cost, f"{pg_time:.4f}", "", "", len(pg_plan), pg_ver])
        print(f"  [3] Paper Learned GBFS     : {'SOLVED ('+str(pg_exp)+' exp, '+str(round(pg_cost,1))+' cost)' if pg_sol else 'TIMEOUT'} in {pg_time*1000:6.1f}ms", flush=True)

        # Setup Hybrid Heuristic Instance
        hybrid_h = ConfidenceAwareMCHeuristic(
            model=model, goal_state=goal_state, box_targets=box_targets,
            device=device, num_mc_samples=5, dropout_p=0.10, lambda_min=0.20, scale_constant_C=50.0, dim=10
        )

        # [4] Fixed 50/50 Hybrid A*
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
        csv_writer.writerow([map_idx, "Fixed 50/50 Hybrid A*", fa_sol, fa_exp, fa_cost, f"{fa_time:.4f}", "0.50", "0.00", len(fa_plan), fa_ver])
        print(f"  [4] Fixed 50/50 Hybrid A*  : {'SOLVED ('+str(fa_exp)+' exp, '+str(round(fa_cost,1))+' cost)' if fa_sol else 'TIMEOUT'} in {fa_time*1000:6.1f}ms", flush=True)

        # Reset trackers
        hybrid_h.classical_tracker.reset()
        for t in hybrid_h.learned_trackers: t.reset()

        # [5] Fixed 50/50 Hybrid GBFS
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
        csv_writer.writerow([map_idx, "Fixed 50/50 Hybrid GBFS", fg_sol, fg_exp, fg_cost, f"{fg_time:.4f}", "0.50", "0.00", len(fg_plan), fg_ver])
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
            map_idx, "Confidence-Aware Hybrid A* (Ours)", ca_sol, ca_exp, ca_cost, f"{ca_time:.4f}",
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
            map_idx, "Confidence-Aware Hybrid GBFS (Ours)", cg_sol, cg_exp, cg_cost, f"{cg_time:.4f}",
            f"{cg_stats['mean_lambda']:.4f}", f"{cg_stats['mean_variance']:.4f}", len(cg_plan), cg_ver
        ])
        csv_file.flush()
        cg_lbl = f"lambda={cg_stats['mean_lambda']:.2f}" if cg_sol else "timed out"
        print(f"  [7] Confidence-Aware GBFS  : {'SOLVED ('+str(cg_exp)+' exp, '+str(round(cg_cost,1))+' cost, '+cg_lbl+')' if cg_sol else 'TIMEOUT'} in {cg_time*1000:6.1f}ms", flush=True)

        # Check periodic statistic logging requirement: every N runs
        if map_idx % args.log_interval == 0:
            print_interim_statistics(summary_data, map_idx, n_instances, total_start, is_final=False)

    csv_file.close()
    total_time = time.time() - total_start

    # Final comprehensive report
    print_interim_statistics(summary_data, n_instances, n_instances, total_start, is_final=True)
    generate_markdown_report(summary_data, n_instances, total_time, args.output_report)
    print(f"Saved full benchmark report to '{args.output_report}' and CSV to '{args.output_csv}'!\n", flush=True)


if __name__ == "__main__":
    main()
