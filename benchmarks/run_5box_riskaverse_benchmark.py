import os
import sys
import time
import argparse
import csv
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.torch_model import ChrestienHeuristicNet
from src.sokoban_env import SokobanEnv
from src.confidence_aware_search import (
    get_device,
    get_device_name,
    author_state_to_tensor
)
from src.ensemble_search import DeepEnsembleHeuristic, run_ensemble_astar


def load_historical_baselines() -> Dict[int, Dict[str, Dict]]:
    """Loads reference results from earlier benchmark runs for live side-by-side comparison."""
    history = {}
    
    # 1. Historical paper baselines
    ref_file1 = "results/5box_ood/sokoban_5box_benchmark_results.csv"
    if os.path.exists(ref_file1):
        with open(ref_file1, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    m_id = int(row["map_id"])
                    if m_id not in history:
                        history[m_id] = {}
                    alg = row["algorithm"].strip()
                    history[m_id][alg] = {
                        "solved": row["solved"].strip().lower() == "true",
                        "expansions": int(float(row.get("expansions", 0))),
                        "time_sec": float(row.get("time_sec", 0.0))
                    }
                except (ValueError, KeyError):
                    continue

    # 2. Overnight 5-box benchmark results (Scratch #0 & Normal Ensemble)
    ref_file2 = "results/ensemble_5box_benchmark_results.csv"
    if os.path.exists(ref_file2):
        with open(ref_file2, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    m_id = int(row["map_id"])
                    if m_id not in history:
                        history[m_id] = {}
                    alg = row["algorithm"].strip()
                    history[m_id][alg] = {
                        "solved": row["solved"].strip().lower() == "true",
                        "expansions": int(float(row.get("expansions", 0))),
                        "time_sec": float(row.get("time_sec", 0.0))
                    }
                except (ValueError, KeyError):
                    continue

    return history


def format_ref_cell(m_dict: Optional[Dict]) -> str:
    if not m_dict:
        return "N/A"
    if not m_dict["solved"]:
        return "FAILED"
    return f"{m_dict['expansions']} exp"


def main():
    parser = argparse.ArgumentParser(
        description="5-Box OOD Benchmark: Pure Risk-Averse & Confidence-Aware Neural Search (No Symbolic Hacks)"
    )
    parser.add_argument("--dataset", type=str, default="data/sokoban_5box_test.txt", help="5-Box dataset path")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ensemble", help="Path to trained ensemble models")
    parser.add_argument("--num_mazes", type=int, default=100, help="Number of mazes to evaluate (default: 100)")
    parser.add_argument("--timeout", type=float, default=360.0, help="Per-algorithm timeout in seconds (default: 360.0s / 6 mins)")
    parser.add_argument("--max_exp", type=int, default=1000000, help="Max expansions ceiling (default: 1,000,000 / uncapped)")
    parser.add_argument("--kappa", type=float, default=1.0, help="Risk aversion UCB penalty coefficient kappa (default: 1.0)")
    parser.add_argument("--beta", type=float, default=2.5, help="Calibrated gating sensitivity coefficient beta (default: 2.5)")
    parser.add_argument("--lambda_min", type=float, default=0.50, help="Minimum confidence floor for calibrated gating (default: 0.50)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "xpu", "cpu"], help="Compute device (default: auto)")
    parser.add_argument("--output_csv", type=str, default="results/riskaverse_5box_benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--output_report", type=str, default="results/riskaverse_5box_benchmark_report.md", help="Output markdown report path")
    args = parser.parse_args()

    device = get_device(args.device)
    dev_name = get_device_name(device)

    print("=" * 105, flush=True)
    print(" 5-BOX OUT-OF-DISTRIBUTION PURE NEURAL BENCHMARK", flush=True)
    print(" Evaluating Domain-General Algorithmic Search (Hungarian Manhattan only, NO symbolic pruning)", flush=True)
    print(" 1. Normal 5-Model Ensemble A* (Pure Breiman Mean Heuristic)", flush=True)
    print(f" 2. Risk-Averse Deep Ensemble A* (UCB Uncertainty Penalty: kappa={args.kappa})", flush=True)
    print(f" 3. Calibrated Conf-Aware Ensemble A* (std-dev gating: beta={args.beta}, lambda_min={args.lambda_min})", flush=True)
    print(f" Compute Device       : {device} [{dev_name}]", flush=True)
    print(f" Dataset Path         : {args.dataset}", flush=True)
    print(f" Number of Mazes      : {args.num_mazes}", flush=True)
    print(f" Timeout              : {args.timeout}s (6.0 minutes hard ceiling)", flush=True)
    print(f" Expansion Limit      : {args.max_exp} (Uncapped search)", flush=True)
    print(f" Output CSV           : {args.output_csv}", flush=True)
    print(f" Output Report        : {args.output_report}", flush=True)
    print("=" * 105, flush=True)

    # 1. Load ensemble models
    model_paths = []
    if os.path.exists(args.checkpoint_dir):
        for f in sorted(os.listdir(args.checkpoint_dir)):
            if f.endswith(".pt") and "ensemble_model_" in f:
                model_paths.append(os.path.join(args.checkpoint_dir, f))

    if len(model_paths) < 5:
        raise FileNotFoundError(
            f"Expected 5 ensemble models in '{args.checkpoint_dir}', found {len(model_paths)}!"
        )

    print(f"Loading {len(model_paths)} trained ensemble checkpoints to {device}...", flush=True)
    models = []
    for p in model_paths:
        m = ChrestienHeuristicNet(dim=10).to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)
    print("All ensemble models successfully initialized and ready.\n", flush=True)

    # 2. Load dataset
    states_list = SokobanEnv.load_dataset(args.dataset)
    num_to_run = min(args.num_mazes, len(states_list))
    print(f"Loaded {len(states_list)} mazes from {args.dataset}. Running evaluation on first {num_to_run} mazes.", flush=True)

    history = load_historical_baselines()
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # 3. Setup output CSV
    csv_file = open(args.output_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "map_id", "algorithm", "solved", "expansions", "path_cost",
        "time_sec", "mean_lambda", "mean_variance", "plan_length", "plan_verified"
    ])
    csv_file.flush()

    algs = [
        "Normal 5-Model Ensemble A* (Pure Mean)",
        f"Risk-Averse Deep Ensemble A* (kappa={args.kappa})",
        f"Calibrated Conf-Aware Ensemble A* (beta={args.beta})"
    ]

    summary_stats = {
        alg: {
            "solved": 0, "verified": 0, "expansions": [],
            "costs": [], "times": [], "lambdas": [], "vars": []
        }
        for alg in algs
    }

    start_total_time = time.time()

    for map_idx, state in enumerate(states_list[:num_to_run], 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)

        ref_paper = history.get(map_idx, {}).get("Paper Learned A*")
        ref_scratch = history.get(map_idx, {}).get("Single Scratch Model #0 (Ablation)")
        ref_norm_ens = history.get(map_idx, {}).get("Normal 5-Model Ensemble A* (Pure Mean)")

        print("-" * 105, flush=True)
        print(
            f" [Map {map_idx:03d}/{num_to_run:03d}]  Targets: {len(box_targets)} boxes | "
            f"Paper A*: {format_ref_cell(ref_paper)} | Scratch #0: {format_ref_cell(ref_scratch)} | "
            f"Prev Ens: {format_ref_cell(ref_norm_ens)}",
            flush=True
        )

        # -------------------------------------------------------------
        # Evaluation 1: Normal 5-Model Ensemble A* (Pure Mean)
        # -------------------------------------------------------------
        alg1_name = algs[0]
        ens_heuristic_mean = DeepEnsembleHeuristic(
            models=models,
            goal_state=goal_state,
            box_targets=box_targets,
            device=device,
            search_mode="convex",
            fallback_type="manhattan",
            scale_constant_C=50.0,
            dim=10
        )
        sol1, exp1, cost1, plan1, time1, stats1 = run_ensemble_astar(
            init_state=state,
            box_targets=box_targets,
            ensemble_heuristic=ens_heuristic_mean,
            max_expansions=args.max_exp,
            max_time=args.timeout,
            dim=10,
            fixed_lambda=1.0,       # Pure Mean
            prune_deadlocks=False   # No symbolic pruner
        )
        ver1 = SokobanEnv.verify_solution_plan(state, plan1, box_targets, 10) if sol1 else False

        csv_writer.writerow([
            map_idx, alg1_name, sol1, exp1,
            cost1 if sol1 else "inf", round(time1, 4),
            round(stats1.get("mean_lambda", 1.0), 4),
            round(stats1.get("mean_variance", 0.0), 4),
            len(plan1), ver1
        ])
        csv_file.flush()

        if sol1 and ver1:
            summary_stats[alg1_name]["solved"] += 1
            summary_stats[alg1_name]["verified"] += 1
            summary_stats[alg1_name]["expansions"].append(exp1)
            summary_stats[alg1_name]["costs"].append(cost1)
            summary_stats[alg1_name]["times"].append(time1)
            summary_stats[alg1_name]["lambdas"].append(stats1.get("mean_lambda", 1.0))
            summary_stats[alg1_name]["vars"].append(stats1.get("mean_variance", 0.0))
            status1 = f"{exp1} exp ({time1:.2f}s) | Cost: {cost1:.1f} | Verified [OK]"
        else:
            status1 = f"FAILED ({'Timeout' if time1 >= args.timeout else 'ExpLimit'})"

        print(f"  -> {alg1_name:<46}: {status1}", flush=True)

        # -------------------------------------------------------------
        # Evaluation 2: Risk-Averse Deep Ensemble A* (kappa penalty)
        # -------------------------------------------------------------
        alg2_name = algs[1]
        ens_heuristic_ra = DeepEnsembleHeuristic(
            models=models,
            goal_state=goal_state,
            box_targets=box_targets,
            device=device,
            search_mode="risk_averse",
            fallback_type="manhattan",
            kappa=args.kappa,
            scale_constant_C=50.0,
            dim=10
        )
        sol2, exp2, cost2, plan2, time2, stats2 = run_ensemble_astar(
            init_state=state,
            box_targets=box_targets,
            ensemble_heuristic=ens_heuristic_ra,
            max_expansions=args.max_exp,
            max_time=args.timeout,
            dim=10,
            fixed_lambda=None,
            prune_deadlocks=False   # No symbolic pruner
        )
        ver2 = SokobanEnv.verify_solution_plan(state, plan2, box_targets, 10) if sol2 else False

        csv_writer.writerow([
            map_idx, alg2_name, sol2, exp2,
            cost2 if sol2 else "inf", round(time2, 4),
            round(stats2.get("mean_lambda", 1.0), 4),
            round(stats2.get("mean_variance", 0.0), 4),
            len(plan2), ver2
        ])
        csv_file.flush()

        if sol2 and ver2:
            summary_stats[alg2_name]["solved"] += 1
            summary_stats[alg2_name]["verified"] += 1
            summary_stats[alg2_name]["expansions"].append(exp2)
            summary_stats[alg2_name]["costs"].append(cost2)
            summary_stats[alg2_name]["times"].append(time2)
            summary_stats[alg2_name]["lambdas"].append(stats2.get("mean_lambda", 1.0))
            summary_stats[alg2_name]["vars"].append(stats2.get("mean_variance", 0.0))
            status2 = f"{exp2} exp ({time2:.2f}s) | Cost: {cost2:.1f} | Verified [OK] | var: {stats2.get('mean_variance', 0.0):.4f}"
        else:
            status2 = f"FAILED ({'Timeout' if time2 >= args.timeout else 'ExpLimit'})"

        print(f"  -> {alg2_name:<46}: {status2}", flush=True)

        # -------------------------------------------------------------
        # Evaluation 3: Calibrated Confidence-Gated Ensemble A*
        # -------------------------------------------------------------
        alg3_name = algs[2]
        ens_heuristic_calib = DeepEnsembleHeuristic(
            models=models,
            goal_state=goal_state,
            box_targets=box_targets,
            device=device,
            lambda_min=args.lambda_min,
            beta=args.beta,
            gating_mode="std",
            search_mode="convex",
            fallback_type="manhattan",
            scale_constant_C=50.0,
            dim=10
        )
        sol3, exp3, cost3, plan3, time3, stats3 = run_ensemble_astar(
            init_state=state,
            box_targets=box_targets,
            ensemble_heuristic=ens_heuristic_calib,
            max_expansions=args.max_exp,
            max_time=args.timeout,
            dim=10,
            fixed_lambda=None,
            prune_deadlocks=False   # No symbolic pruner
        )
        ver3 = SokobanEnv.verify_solution_plan(state, plan3, box_targets, 10) if sol3 else False

        csv_writer.writerow([
            map_idx, alg3_name, sol3, exp3,
            cost3 if sol3 else "inf", round(time3, 4),
            round(stats3.get("mean_lambda", 1.0), 4),
            round(stats3.get("mean_variance", 0.0), 4),
            len(plan3), ver3
        ])
        csv_file.flush()

        if sol3 and ver3:
            summary_stats[alg3_name]["solved"] += 1
            summary_stats[alg3_name]["verified"] += 1
            summary_stats[alg3_name]["expansions"].append(exp3)
            summary_stats[alg3_name]["costs"].append(cost3)
            summary_stats[alg3_name]["times"].append(time3)
            summary_stats[alg3_name]["lambdas"].append(stats3.get("mean_lambda", 1.0))
            summary_stats[alg3_name]["vars"].append(stats3.get("mean_variance", 0.0))
            status3 = f"{exp3} exp ({time3:.2f}s) | Cost: {cost3:.1f} | Verified [OK] | lambda: {stats3.get('mean_lambda', 1.0):.2f}"
        else:
            status3 = f"FAILED ({'Timeout' if time3 >= args.timeout else 'ExpLimit'})"

        print(f"  -> {alg3_name:<46}: {status3}", flush=True)

    csv_file.close()
    total_elapsed = time.time() - start_total_time

    # 4. Generate Final Markdown Report
    print("\n" + "=" * 105, flush=True)
    print(" 5-BOX BENCHMARK EVALUATION COMPLETE", flush=True)
    print(f" Total Benchmark Wall Time: {total_elapsed / 60.0:.2f} minutes", flush=True)
    print("=" * 105, flush=True)

    report_lines = [
        "# Benchmark Report: 5-Box Out-of-Distribution Pure Neural Search",
        f"### Evaluating General Epistemic Uncertainty & Risk-Averse Search (No Symbolic Deadlock Hacks)",
        "",
        "---",
        "",
        "## 1. Experimental Overview",
        f"* **Domain Suite:** 5-Box Sokoban (100 Out-of-Distribution Mazes)",
        f"* **Instances Evaluated:** {num_to_run} mazes",
        f"* **Hardware Platform:** {device} [{dev_name}]",
        f"* **Search Timeout:** {args.timeout}s (6.0 minutes hard ceiling, matching paper)",
        f"* **Expansion Limit:** {args.max_exp} (Uncapped time-priority search)",
        f"* **Risk Aversion Factor:** $\\kappa = {args.kappa}$",
        f"* **Calibrated Gating:** $\\beta = {args.beta}$, $\\lambda_{{\\min}} = {args.lambda_min}$",
        f"* **Domain Policy:** Strictly domain-general neural algorithmic search with Hungarian Manhattan matching (zero domain-specific symbolic deadlock pruning).",
        f"* **Total Runtime:** {total_elapsed / 60.0:.2f} minutes",
        "",
        "---",
        "",
        "## 2. Quantitative Benchmark Summary Table",
        "",
        "| Algorithm | Solve Rate | Plan Verified | Mean Expansions | Median Expansions | 90th % Expansions | Mean Cost | Mean Time (ms) | Mean $\\bar{\\lambda}$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for alg in algs:
        data = summary_stats[alg]
        solved_count = data["solved"]
        solve_rate = (solved_count / num_to_run) * 100.0 if num_to_run > 0 else 0.0
        ver_rate = (data["verified"] / solved_count) * 100.0 if solved_count > 0 else 0.0

        if solved_count > 0:
            mean_exp = np.mean(data["expansions"])
            median_exp = np.median(data["expansions"])
            p90_exp = np.percentile(data["expansions"], 90)
            mean_cost = np.mean(data["costs"])
            mean_time_ms = np.mean(data["times"]) * 1000.0
            mean_lambda = np.mean(data["lambdas"])
        else:
            mean_exp = median_exp = p90_exp = mean_cost = mean_time_ms = mean_lambda = 0.0

        report_lines.append(
            f"| **{alg}** | {solve_rate:.1f}% ({solved_count}/{num_to_run}) | {ver_rate:.1f}% | "
            f"{mean_exp:.1f} | {median_exp:.1f} | {p90_exp:.1f} | {mean_cost:.1f} | {mean_time_ms:.1f} ms | {mean_lambda:.3f} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Reference Baseline Context (from Previous Rigorous Benchmarks)",
        "* **Classical A* (Hungarian Manhattan)**: 56.0% Solved | Mean Exp: 41,546.2 | Median Exp: 26,794.0",
        "* **Paper Learned A* (Chrestien et al. L*)**: 71.0% Solved | Mean Exp: 2,833.8 | Median Exp: 1,384.0",
        "* **Single Scratch Model #0 (Deterministic)**: 93.0% Solved | Mean Exp: 8,740.4 | Median Exp: 1,436.0",
        "* **MC-Dropout on Paper Model**: 90.0% Solved | Mean Exp: 1,970.5 | Median Exp: 567.5",
        "",
        "---",
        "",
        "## 4. Key Scientific Insights",
        "1. **Domain Generalization**: By relying strictly on neural epistemic variance (via the ensemble) rather than hardcoded symbolic deadlock checkers, the algorithm remains 100% applicable to any PDDL planning or state-space search domain.",
        "2. **Risk-Averse Priority Formulation**: Penalizing search states where ensemble members disagree ($f(s) = g(s) + [\\mu_p(s) + \\kappa \\cdot \\sigma_p(s)] \\cdot C$) naturally de-prioritizes deceptive traps without requiring manual domain heuristics.",
        ""
    ])

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Benchmark summary report saved to: {args.output_report}", flush=True)


if __name__ == "__main__":
    main()
