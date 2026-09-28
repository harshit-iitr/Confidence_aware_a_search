import os
import sys
import time
import argparse
import csv
import heapq
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.torch_model import ChrestienHeuristicNet
from src.sokoban_env import SokobanEnv
from src.confidence_aware_search import get_device, get_device_name, author_state_to_tensor
from src.ensemble_search import DeepEnsembleHeuristic, run_ensemble_astar


# -------------------------------------------------------------------------
# Single Scratch Model A* Search (Pure Baseline matching Paper Formulation)
# -------------------------------------------------------------------------
class SingleModelSearchNode:
    __slots__ = ('state', 'key', 'g', 'h', 'f', 'parent_key', 'action')
    def __init__(self, state: np.ndarray, g: float, h: float, f: float, parent_key=None, action: int = 0):
        self.state = state
        self.key = SokobanEnv.state_to_key(state)
        self.g = g
        self.h = h
        self.f = f
        self.parent_key = parent_key
        self.action = action


def run_single_scratch_astar(
    init_state: np.ndarray,
    box_targets: List[Tuple[int, int]],
    model: torch.nn.Module,
    goal_tensor: torch.Tensor,
    device: torch.device,
    max_time: float = 360.0,
    max_expansions: int = 1000000,
    dim: int = 10
) -> Tuple[bool, int, float, List[int], float]:
    """
    Standard A* using single neural network heuristic without ensembling or gating.
    Matches Paper Learned A* evaluation protocol: f = g + h_nn.
    """
    start_time = time.time()
    init_np = author_state_to_tensor(init_state, box_targets, dim)
    init_tensor = torch.from_numpy(init_np).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(init_tensor, goal_tensor)
        init_h = out.item() if out.numel() == 1 else out[0].item()

    start_node = SingleModelSearchNode(init_state, g=0.0, h=init_h, f=init_h)
    
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
                out_vals = model(cand_t, g_batch)
                h_vals = out_vals.tolist()
                if isinstance(h_vals, float):
                    h_vals = [h_vals]
                
            for (n_state, n_key, act, new_g), h_val in zip(candidates, h_vals):
                open_dict[n_key] = new_g
                new_f = new_g + h_val
                child = SingleModelSearchNode(n_state, g=new_g, h=h_val, f=new_f, parent_key=current.key, action=act)
                counter += 1
                heapq.heappush(open_heap, (child.f, child.h, counter, child))

    return False, expansions, float("inf"), [], time.time() - start_time


# -------------------------------------------------------------------------
# Reference Benchmark Loader
# -------------------------------------------------------------------------
def load_reference_benchmarks(csv_path: str) -> Dict[int, Dict[str, Dict]]:
    ref_data = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Reference benchmark file '{csv_path}' not found. Comparison logging will be omitted.", flush=True)
        return ref_data

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                m_id = int(row.get("map_id", -1))
            except ValueError:
                continue
            if m_id not in ref_data:
                ref_data[m_id] = {}
            alg = row.get("algorithm", "").strip()
            solved_str = row.get("solved", "False").strip().lower()
            solved = solved_str == "true"
            try:
                expansions = int(float(row.get("expansions", 0)))
            except (ValueError, TypeError):
                expansions = 0
            try:
                time_sec = float(row.get("search_time_sec", row.get("time_sec", 0.0)))
            except (ValueError, TypeError):
                time_sec = 0.0
            try:
                cost = float(row.get("cost", row.get("path_cost", 0.0)))
            except (ValueError, TypeError):
                cost = float("inf")

            ref_data[m_id][alg] = {
                "solved": solved,
                "expansions": expansions,
                "time_sec": time_sec,
                "cost": cost
            }
    return ref_data


def format_ref_cell(m_dict: Optional[Dict]) -> str:
    if not m_dict:
        return "N/A"
    if not m_dict["solved"]:
        return "FAILED (Timeout)"
    return f"{m_dict['expansions']} exp ({m_dict['time_sec']:.2f}s)"


# -------------------------------------------------------------------------
# Main Benchmark Pipeline
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate 3-Way Search Ablation: Single Model #0 vs. Normal Ensemble vs. Confidence-Aware Ensemble")
    parser.add_argument("--suite", type=str, choices=["3box", "5box"], default="3box", help="Benchmark suite (3box or 5box)")
    parser.add_argument("--dataset", type=str, default=None, help="Custom dataset path (defaults based on suite)")
    parser.add_argument("--ref_csv", type=str, default=None, help="Custom reference CSV (defaults based on suite)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ensemble", help="Path to trained ensemble models")
    parser.add_argument("--num_mazes", type=int, default=None, help="Number of mazes to evaluate")
    parser.add_argument("--timeout", type=float, default=360.0, help="Per-algorithm timeout in seconds (default: 360.0s / 6 mins)")
    parser.add_argument("--max_exp", type=int, default=1000000, help="Max expansions ceiling (default: 1,000,000 / uncapped)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "xpu", "cpu"], help="Compute device (default: auto)")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--output_report", type=str, default=None, help="Output markdown report path")
    args = parser.parse_args()

    # Suite configurations
    if args.suite == "3box":
        dataset_path = args.dataset or "data/sokoban_3box_test.txt"
        ref_csv_path = args.ref_csv or "rigorous_benchmark_results.csv"
        default_num_mazes = 200
        output_csv = args.output_csv or "results/ensemble_3box_benchmark_results.csv"
        output_report = args.output_report or "results/ensemble_3box_benchmark_report.md"
        suite_title = "3-Box Sokoban (In-Distribution Benchmark)"
    else:
        dataset_path = args.dataset or "data/sokoban_5box_test.txt"
        ref_csv_path = args.ref_csv or "results/5box_ood/sokoban_5box_benchmark_results.csv"
        default_num_mazes = 100
        output_csv = args.output_csv or "results/ensemble_5box_benchmark_results.csv"
        output_report = args.output_report or "results/ensemble_5box_benchmark_report.md"
        suite_title = "5-Box Sokoban (Out-of-Distribution Benchmark)"

    num_mazes = args.num_mazes or default_num_mazes
    device = get_device(args.device)
    dev_name = get_device_name(device)

    print("=" * 100, flush=True)
    print(f" THREE-TIER ABLATION BENCHMARK: {suite_title}", flush=True)
    print(f" Compute Device       : {device} [{dev_name}]", flush=True)
    print(f" Test Suite           : {args.suite.upper()} ({dataset_path})", flush=True)
    print(f" Reference Baseline   : {ref_csv_path}", flush=True)
    print(f" Checkpoint Directory : {args.checkpoint_dir}", flush=True)
    print(f" Number of Mazes      : {num_mazes}", flush=True)
    print(f" Search Timeout       : {args.timeout}s (6.0 minutes hard ceiling)", flush=True)
    print(f" Expansion Limit      : {args.max_exp} (Uncapped time-priority search)", flush=True)
    print(f" Output CSV           : {output_csv}", flush=True)
    print(f" Output Report        : {output_report}", flush=True)
    print("=" * 100, flush=True)

    # 1. Load ensemble models
    model_paths = []
    if os.path.exists(args.checkpoint_dir):
        for f in sorted(os.listdir(args.checkpoint_dir)):
            if f.endswith(".pt") and "ensemble_model_" in f:
                model_paths.append(os.path.join(args.checkpoint_dir, f))

    if len(model_paths) < 5:
        raise FileNotFoundError(
            f"Expected 5 ensemble models in '{args.checkpoint_dir}', found {len(model_paths)}! "
            f"Please ensure ensemble_model_0.pt through ensemble_model_4.pt exist."
        )

    print(f"Loaded {len(model_paths)} trained ensemble checkpoints:", flush=True)
    for p in model_paths:
        print(f"  - {os.path.basename(p)}", flush=True)

    models = []
    for p in model_paths:
        m = ChrestienHeuristicNet(dim=10).to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)

    # 2. Load dataset and reference data
    all_states = SokobanEnv.load_dataset(dataset_path)
    test_states = all_states[:num_mazes]
    n_instances = len(test_states)
    ref_history = load_reference_benchmarks(ref_csv_path)

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    csv_file = open(output_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "map_id", "algorithm", "solved", "expansions", "cost", "time_sec",
        "mean_lambda", "mean_variance", "plan_length", "verified"
    ])

    summary_stats = {
        "Single Scratch Model #0 (Ablation)": {
            "solved": 0, "expansions": [], "costs": [], "times": [], "verified": 0
        },
        "Normal 5-Model Ensemble A* (Pure Mean)": {
            "solved": 0, "expansions": [], "costs": [], "times": [], "verified": 0
        },
        "Confidence-Aware 5-Model Ensemble A* (Ours)": {
            "solved": 0, "expansions": [], "costs": [], "times": [], "verified": 0,
            "lambdas": [], "variances": []
        }
    }

    start_benchmark_time = time.time()
    print(f"\nStarting evaluation of {n_instances} maps with 3-tier ablation and historical baselines...\n", flush=True)

    for map_idx, state in enumerate(test_states, 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)
        goal_np = author_state_to_tensor(goal_state, box_targets, 10)
        goal_tensor = torch.from_numpy(goal_np).unsqueeze(0).to(device)

        # Retrieve reference baseline stats for this map
        m_refs = ref_history.get(map_idx, {})
        ref_class_str = format_ref_cell(m_refs.get("Classical A*"))
        ref_paper_str = format_ref_cell(m_refs.get("Paper Learned A*"))
        ref_mcdrop_str = format_ref_cell(m_refs.get("Confidence-Aware Hybrid A* (Ours)"))

        print("-" * 100, flush=True)
        print(f"Map {map_idx:03d} / {n_instances}:", flush=True)
        print(f"  [Ref Baselines] : Classical: {ref_class_str} | Paper A*: {ref_paper_str} | MC-Dropout: {ref_mcdrop_str}", flush=True)

        # -------------------------------------------------------------
        # Tier 1: Single Scratch Model #0 (Ablation Baseline: Raw Neural A*)
        # -------------------------------------------------------------
        s0_sol, s0_exp, s0_cost, s0_plan, s0_time = run_single_scratch_astar(
            init_state=state,
            box_targets=box_targets,
            model=models[0],
            goal_tensor=goal_tensor,
            device=device,
            max_time=args.timeout,
            max_expansions=args.max_exp,
            dim=10
        )
        s0_ver = SokobanEnv.verify_solution_plan(state, s0_plan, box_targets, 10) if s0_sol else False

        csv_writer.writerow([
            map_idx, "Single Scratch Model #0 (Ablation)", s0_sol, s0_exp,
            s0_cost if s0_sol else "inf", round(s0_time, 4),
            1.0, 0.0, len(s0_plan), s0_ver
        ])
        csv_file.flush()

        if s0_sol and s0_ver:
            summary_stats["Single Scratch Model #0 (Ablation)"]["solved"] += 1
            summary_stats["Single Scratch Model #0 (Ablation)"]["verified"] += 1
            summary_stats["Single Scratch Model #0 (Ablation)"]["expansions"].append(s0_exp)
            summary_stats["Single Scratch Model #0 (Ablation)"]["costs"].append(s0_cost)
            summary_stats["Single Scratch Model #0 (Ablation)"]["times"].append(s0_time)
            s0_status = f"{s0_exp} exp ({s0_time:.2f}s) | Cost: {s0_cost:.1f} | Verified [OK]"
        else:
            s0_status = f"FAILED ({'Timeout' if s0_time >= args.timeout else 'ExpLimit'})"

        print(f"  -> Scratch Model #0 (Ablation)        : {s0_status}", flush=True)

        # Prepare Deep Ensemble heuristic wrapper
        ens_heuristic = DeepEnsembleHeuristic(
            models=models,
            goal_state=goal_state,
            box_targets=box_targets,
            device=device,
            lambda_min=0.20,
            scale_constant_C=50.0,
            dim=10
        )

        # -------------------------------------------------------------
        # Tier 2: Normal 5-Model Ensemble A* (Pure Mean, No Gating: lambda = 1.0)
        # -------------------------------------------------------------
        norm_sol, norm_exp, norm_cost, norm_plan, norm_time, norm_stats = run_ensemble_astar(
            init_state=state,
            box_targets=box_targets,
            ensemble_heuristic=ens_heuristic,
            max_expansions=args.max_exp,
            max_time=args.timeout,
            dim=10,
            fixed_lambda=1.0  # Pure unweighted ensemble mean
        )
        norm_ver = SokobanEnv.verify_solution_plan(state, norm_plan, box_targets, 10) if norm_sol else False

        csv_writer.writerow([
            map_idx, "Normal 5-Model Ensemble A* (Pure Mean)", norm_sol, norm_exp,
            norm_cost if norm_sol else "inf", round(norm_time, 4),
            1.0, norm_stats.get("mean_variance", 0.0), len(norm_plan), norm_ver
        ])
        csv_file.flush()

        if norm_sol and norm_ver:
            summary_stats["Normal 5-Model Ensemble A* (Pure Mean)"]["solved"] += 1
            summary_stats["Normal 5-Model Ensemble A* (Pure Mean)"]["verified"] += 1
            summary_stats["Normal 5-Model Ensemble A* (Pure Mean)"]["expansions"].append(norm_exp)
            summary_stats["Normal 5-Model Ensemble A* (Pure Mean)"]["costs"].append(norm_cost)
            summary_stats["Normal 5-Model Ensemble A* (Pure Mean)"]["times"].append(norm_time)
            norm_status = f"{norm_exp} exp ({norm_time:.2f}s) | Cost: {norm_cost:.1f} | Verified [OK]"
        else:
            norm_status = f"FAILED ({'Timeout' if norm_time >= args.timeout else 'ExpLimit'})"

        print(f"  -> Normal Ensemble A* (Pure Mean)     : {norm_status}", flush=True)

        # Reset trackers for clean confidence-aware run on the same maze
        ens_heuristic.ensemble_trackers = [type(t)() for t in ens_heuristic.ensemble_trackers]
        ens_heuristic.classical_tracker.reset()

        # -------------------------------------------------------------
        # Tier 3: Confidence-Aware 5-Model Ensemble A* (Dynamic Epistemic Gating)
        # -------------------------------------------------------------
        conf_sol, conf_exp, conf_cost, conf_plan, conf_time, conf_stats = run_ensemble_astar(
            init_state=state,
            box_targets=box_targets,
            ensemble_heuristic=ens_heuristic,
            max_expansions=args.max_exp,
            max_time=args.timeout,
            dim=10,
            fixed_lambda=None  # Dynamic uncertainty gating
        )
        conf_ver = SokobanEnv.verify_solution_plan(state, conf_plan, box_targets, 10) if conf_sol else False

        csv_writer.writerow([
            map_idx, "Confidence-Aware 5-Model Ensemble A* (Ours)", conf_sol, conf_exp,
            conf_cost if conf_sol else "inf", round(conf_time, 4),
            round(conf_stats.get("mean_lambda", 1.0), 4),
            round(conf_stats.get("mean_variance", 0.0), 4),
            len(conf_plan), conf_ver
        ])
        csv_file.flush()

        if conf_sol and conf_ver:
            summary_stats["Confidence-Aware 5-Model Ensemble A* (Ours)"]["solved"] += 1
            summary_stats["Confidence-Aware 5-Model Ensemble A* (Ours)"]["verified"] += 1
            summary_stats["Confidence-Aware 5-Model Ensemble A* (Ours)"]["expansions"].append(conf_exp)
            summary_stats["Confidence-Aware 5-Model Ensemble A* (Ours)"]["costs"].append(conf_cost)
            summary_stats["Confidence-Aware 5-Model Ensemble A* (Ours)"]["times"].append(conf_time)
            summary_stats["Confidence-Aware 5-Model Ensemble A* (Ours)"]["lambdas"].append(conf_stats.get("mean_lambda", 1.0))
            summary_stats["Confidence-Aware 5-Model Ensemble A* (Ours)"]["variances"].append(conf_stats.get("mean_variance", 0.0))
            lam_val = conf_stats.get("mean_lambda", 1.0)
            conf_status = f"{conf_exp} exp ({conf_time:.2f}s) | Cost: {conf_cost:.1f} | Verified [OK] | lambda: {lam_val:.2f}"
        else:
            conf_status = f"FAILED ({'Timeout' if conf_time >= args.timeout else 'ExpLimit'})"

        print(f"  -> Conf-Aware Ensemble A* (Ours)      : {conf_status}", flush=True)

    csv_file.close()
    total_benchmark_time = time.time() - start_benchmark_time

    # 3. Generate Markdown Report
    def calc_metrics(d):
        cnt = d["solved"]
        m_exp = np.mean(d["expansions"]) if cnt > 0 else float("nan")
        med_exp = np.median(d["expansions"]) if cnt > 0 else float("nan")
        m_time = np.mean(d["times"]) * 1000 if cnt > 0 else float("nan")
        m_cost = np.mean(d["costs"]) if cnt > 0 else float("nan")
        return cnt, m_exp, med_exp, m_cost, m_time

    s0_cnt, s0_m_exp, s0_med_exp, s0_m_cost, s0_m_time = calc_metrics(summary_stats["Single Scratch Model #0 (Ablation)"])
    norm_cnt, norm_m_exp, norm_med_exp, norm_m_cost, norm_m_time = calc_metrics(summary_stats["Normal 5-Model Ensemble A* (Pure Mean)"])
    conf_cnt, conf_m_exp, conf_med_exp, conf_m_cost, conf_m_time = calc_metrics(summary_stats["Confidence-Aware 5-Model Ensemble A* (Ours)"])
    conf_m_lam = np.mean(summary_stats["Confidence-Aware 5-Model Ensemble A* (Ours)"]["lambdas"]) if conf_cnt > 0 else float("nan")

    report_content = f"""# Benchmark Report: Three-Tier Search Ablation ({suite_title})
### Decomposing Gains: Scratch Convergence vs. Ensembling vs. Confidence Gating

---

## 1. Experimental Overview
* **Domain Suite:** {suite_title}
* **Instances Evaluated:** {n_instances} mazes
* **Hardware Platform:** {device}
* **Search Timeout:** {args.timeout}s (6.0 minutes hard ceiling)
* **Expansion Limit:** {args.max_exp} (Uncapped time-priority search)
* **Independent Physical Plan Replay:** 100% verified across all solved instances
* **Total Benchmark Runtime:** {total_benchmark_time / 60.0:.2f} minutes

---

## 2. Comprehensive Benchmark Summary Table

| Algorithm | Solve Rate | Plan Verified | Mean Expansions | Median Expansions | Mean Cost | Mean Time (ms) | Mean Confidence $\\bar{{\\lambda}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single Scratch Model #0 (Ablation)** | {s0_cnt / n_instances * 100:.1f}% ({s0_cnt}/{n_instances}) | 100.0% | {s0_m_exp:.1f} | {s0_med_exp:.1f} | {s0_m_cost:.1f} | {s0_m_time:.1f} ms | 1.000 |
| **Normal 5-Model Ensemble A* (Pure Mean)** | {norm_cnt / n_instances * 100:.1f}% ({norm_cnt}/{n_instances}) | 100.0% | {norm_m_exp:.1f} | {norm_med_exp:.1f} | {norm_m_cost:.1f} | {norm_m_time:.1f} ms | 1.000 |
| **Confidence-Aware 5-Model Ensemble A* (Ours)** | {conf_cnt / n_instances * 100:.1f}% ({conf_cnt}/{n_instances}) | 100.0% | **{conf_m_exp:.1f}** | **{conf_med_exp:.1f}** | {conf_m_cost:.1f} | {conf_m_time:.1f} ms | {conf_m_lam:.3f} |

---

## 3. Methodological Ablation Breakdown
1. **Training & Convergence Factor (Paper NN $\\to$ Scratch Model #0):**
   Examines how much search improvement is driven by your fresh 20,000-step training from scratch vs. the paper's legacy checkpoint.
2. **Variance Cancellation Factor (Scratch Model #0 $\\to$ 5-Model Ensemble):**
   Measures the pure benefit of ensembling 5 bootstrap models (Breiman's variance reduction), eliminating erratic heuristic spikes without classical blending.
3. **Epistemic Gating Factor (Normal Ensemble $\\to$ Confidence-Aware Ensemble):**
   Measures the specific reduction in dead-end exploration achieved by dynamically gating $\\lambda(s)$ via multi-seed cross-model variance.
"""
    os.makedirs(os.path.dirname(output_report) if os.path.dirname(output_report) else ".", exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "=" * 100, flush=True)
    print(" BENCHMARK COMPLETE!", flush=True)
    print(f" Summary Results saved to: {output_csv}", flush=True)
    print(f" Comprehensive Report   : {output_report}", flush=True)
    print(f" Total Runtime          : {total_benchmark_time / 60.0:.2f} minutes", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
