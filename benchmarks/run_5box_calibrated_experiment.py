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
    author_state_to_tensor,
    ConfidenceAwareMCHeuristic,
    run_confidence_aware_astar
)
from src.ensemble_search import DeepEnsembleHeuristic, run_ensemble_astar


def load_historical_baselines() -> Dict[int, Dict[str, Dict]]:
    history = {}
    
    # 1. Historical paper & MC-dropout results
    ref_file1 = "results/5box_ood/sokoban_5box_benchmark_results.csv"
    if os.path.exists(ref_file1):
        with open(ref_file1, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    m_id = int(row["map_id"])
                    if m_id not in history: history[m_id] = {}
                    alg = row["algorithm"].strip()
                    history[m_id][alg] = {
                        "solved": row["solved"].strip().lower() == "true",
                        "expansions": int(float(row.get("expansions", 0))),
                        "time_sec": float(row.get("time_sec", 0.0))
                    }
                except (ValueError, KeyError):
                    continue

    # 2. Overnight ensemble results (Scratch #0 deterministic & Normal ensemble)
    ref_file2 = "results/ensemble_5box_benchmark_results.csv"
    if os.path.exists(ref_file2):
        with open(ref_file2, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    m_id = int(row["map_id"])
                    if m_id not in history: history[m_id] = {}
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
    parser = argparse.ArgumentParser(description="5-Box Focus: MC-Dropout on Scratch #0 vs. Calibrated Ensemble Gating")
    parser.add_argument("--dataset", type=str, default="data/sokoban_5box_test.txt", help="5-Box dataset path")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ensemble", help="Path to trained ensemble models")
    parser.add_argument("--num_mazes", type=int, default=100, help="Number of mazes to evaluate")
    parser.add_argument("--timeout", type=float, default=360.0, help="Per-algorithm timeout in seconds (default: 360.0s / 6 mins)")
    parser.add_argument("--max_exp", type=int, default=1000000, help="Max expansions ceiling (default: 1,000,000 / uncapped)")
    parser.add_argument("--beta", type=float, default=2.5, help="Standard deviation sensitivity coefficient (default: 2.5)")
    parser.add_argument("--lambda_min", type=float, default=0.50, help="Minimum confidence floor (default: 0.50)")
    parser.add_argument("--output_csv", type=str, default="results/calibrated_5box_experiment_results.csv", help="Output CSV path")
    parser.add_argument("--output_report", type=str, default="results/calibrated_5box_experiment_report.md", help="Output markdown report path")
    args = parser.parse_args()

    device = get_device()
    print("=" * 105, flush=True)
    print(" 5-BOX OUT-OF-DISTRIBUTION FOCUSED EXPERIMENT", flush=True)
    print(" 1. MC-Dropout applied to Scratch Model #0 (Ablation against Paper MC-Dropout)", flush=True)
    print(" 2. Calibrated Gating on 5-Model Ensemble (std-dev based, beta=2.5, lambda_min=0.50)", flush=True)
    print(f" Compute Device       : {device}", flush=True)
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
        raise FileNotFoundError(f"Expected 5 ensemble models in '{args.checkpoint_dir}', found {len(model_paths)}!")

    print(f"Loaded {len(model_paths)} trained ensemble checkpoints for device {device}:", flush=True)
    models = []
    for p in model_paths:
        m = ChrestienHeuristicNet(dim=10).to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)

    # 2. Load dataset and reference data
    all_states = SokobanEnv.load_dataset(args.dataset)
    test_states = all_states[:args.num_mazes]
    n_instances = len(test_states)
    history = load_historical_baselines()

    os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else ".", exist_ok=True)
    csv_file = open(args.output_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "map_id", "algorithm", "solved", "expansions", "cost", "time_sec",
        "mean_lambda", "mean_variance", "plan_length", "verified"
    ])

    summary_stats = {
        "MC-Dropout (Scratch Model #0)": {
            "solved": 0, "expansions": [], "costs": [], "times": [], "verified": 0, "lambdas": [], "vars": []
        },
        "Calibrated Conf-Aware Ensemble (Ours)": {
            "solved": 0, "expansions": [], "costs": [], "times": [], "verified": 0, "lambdas": [], "vars": []
        }
    }

    start_benchmark_time = time.time()
    print(f"\nStarting 5-Box evaluation of {n_instances} maps...\n", flush=True)

    for map_idx, state in enumerate(test_states, 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)

        # Retrieve reference baselines for logging
        m_refs = history.get(map_idx, {})
        ref_paper = format_ref_cell(m_refs.get("Paper Learned A*"))
        ref_mc_paper = format_ref_cell(m_refs.get("Confidence-Aware Hybrid A* (Ours)"))
        ref_s0_det = format_ref_cell(m_refs.get("Single Scratch Model #0 (Ablation)"))
        ref_norm_ens = format_ref_cell(m_refs.get("Normal 5-Model Ensemble A* (Pure Mean)"))

        print("-" * 105, flush=True)
        print(f"Map {map_idx:03d} / {n_instances}:", flush=True)
        print(f"  [Ref Baselines] : Paper A*: {ref_paper} | MC-Drop (Paper): {ref_mc_paper} | Scratch #0 Det: {ref_s0_det} | Normal Ens: {ref_norm_ens}", flush=True)

        # -------------------------------------------------------------
        # Experiment 1: MC-Dropout on Scratch Model #0
        # -------------------------------------------------------------
        mc_heuristic = ConfidenceAwareMCHeuristic(
            model=models[0],
            goal_state=goal_state,
            box_targets=box_targets,
            device=device,
            num_mc_samples=5,
            dropout_p=0.10,
            lambda_min=0.20,
            scale_constant_C=50.0,
            dim=10
        )
        mc_sol, mc_exp, mc_cost, mc_plan, mc_time, mc_stats = run_confidence_aware_astar(
            init_state=state,
            box_targets=box_targets,
            hybrid_heuristic=mc_heuristic,
            max_expansions=args.max_exp,
            max_time=args.timeout,
            dim=10
        )
        mc_ver = SokobanEnv.verify_solution_plan(state, mc_plan, box_targets, 10) if mc_sol else False

        csv_writer.writerow([
            map_idx, "MC-Dropout (Scratch Model #0)", mc_sol, mc_exp,
            mc_cost if mc_sol else "inf", round(mc_time, 4),
            round(mc_stats.get("mean_lambda", 1.0), 4),
            round(mc_stats.get("mean_variance", 0.0), 4),
            len(mc_plan), mc_ver
        ])
        csv_file.flush()

        if mc_sol and mc_ver:
            summary_stats["MC-Dropout (Scratch Model #0)"]["solved"] += 1
            summary_stats["MC-Dropout (Scratch Model #0)"]["verified"] += 1
            summary_stats["MC-Dropout (Scratch Model #0)"]["expansions"].append(mc_exp)
            summary_stats["MC-Dropout (Scratch Model #0)"]["costs"].append(mc_cost)
            summary_stats["MC-Dropout (Scratch Model #0)"]["times"].append(mc_time)
            summary_stats["MC-Dropout (Scratch Model #0)"]["lambdas"].append(mc_stats.get("mean_lambda", 1.0))
            summary_stats["MC-Dropout (Scratch Model #0)"]["vars"].append(mc_stats.get("mean_variance", 0.0))
            mc_status = f"{mc_exp} exp ({mc_time:.2f}s) | Cost: {mc_cost:.1f} | Verified [OK] | lambda: {mc_stats.get('mean_lambda', 1.0):.2f}"
        else:
            mc_status = f"FAILED ({'Timeout' if mc_time >= args.timeout else 'ExpLimit'})"

        print(f"  -> MC-Dropout (Scratch #0)             : {mc_status}", flush=True)

        # -------------------------------------------------------------
        # Experiment 2: Calibrated Confidence-Aware 5-Model Ensemble
        # -------------------------------------------------------------
        calib_heuristic = DeepEnsembleHeuristic(
            models=models,
            goal_state=goal_state,
            box_targets=box_targets,
            device=device,
            lambda_min=args.lambda_min,
            beta=args.beta,
            gating_mode="std",
            scale_constant_C=50.0,
            dim=10
        )
        ens_sol, ens_exp, ens_cost, ens_plan, ens_time, ens_stats = run_ensemble_astar(
            init_state=state,
            box_targets=box_targets,
            ensemble_heuristic=calib_heuristic,
            max_expansions=args.max_exp,
            max_time=args.timeout,
            dim=10,
            fixed_lambda=None  # Calibrated dynamic gating
        )
        ens_ver = SokobanEnv.verify_solution_plan(state, ens_plan, box_targets, 10) if ens_sol else False

        csv_writer.writerow([
            map_idx, "Calibrated Conf-Aware Ensemble (Ours)", ens_sol, ens_exp,
            ens_cost if ens_sol else "inf", round(ens_time, 4),
            round(ens_stats.get("mean_lambda", 1.0), 4),
            round(ens_stats.get("mean_variance", 0.0), 4),
            len(ens_plan), ens_ver
        ])
        csv_file.flush()

        if ens_sol and ens_ver:
            summary_stats["Calibrated Conf-Aware Ensemble (Ours)"]["solved"] += 1
            summary_stats["Calibrated Conf-Aware Ensemble (Ours)"]["verified"] += 1
            summary_stats["Calibrated Conf-Aware Ensemble (Ours)"]["expansions"].append(ens_exp)
            summary_stats["Calibrated Conf-Aware Ensemble (Ours)"]["costs"].append(ens_cost)
            summary_stats["Calibrated Conf-Aware Ensemble (Ours)"]["times"].append(ens_time)
            summary_stats["Calibrated Conf-Aware Ensemble (Ours)"]["lambdas"].append(ens_stats.get("mean_lambda", 1.0))
            summary_stats["Calibrated Conf-Aware Ensemble (Ours)"]["vars"].append(ens_stats.get("mean_variance", 0.0))
            ens_status = f"{ens_exp} exp ({ens_time:.2f}s) | Cost: {ens_cost:.1f} | Verified [OK] | lambda: {ens_stats.get('mean_lambda', 1.0):.2f}"
        else:
            ens_status = f"FAILED ({'Timeout' if ens_time >= args.timeout else 'ExpLimit'})"

        print(f"  -> Calibrated Conf-Aware Ensemble (Ours): {ens_status}", flush=True)

    csv_file.close()
    total_benchmark_time = time.time() - start_benchmark_time

    # 3. Generate Markdown Report
    def calc_metrics(d):
        cnt = d["solved"]
        m_exp = np.mean(d["expansions"]) if cnt > 0 else float("nan")
        med_exp = np.median(d["expansions"]) if cnt > 0 else float("nan")
        p90_exp = np.percentile(d["expansions"], 90) if cnt > 0 else float("nan")
        m_time = np.mean(d["times"]) * 1000 if cnt > 0 else float("nan")
        m_cost = np.mean(d["costs"]) if cnt > 0 else float("nan")
        m_lam = np.mean(d["lambdas"]) if cnt > 0 else float("nan")
        return cnt, m_exp, med_exp, p90_exp, m_cost, m_time, m_lam

    mc_cnt, mc_m_exp, mc_med_exp, mc_p90_exp, mc_m_cost, mc_m_time, mc_m_lam = calc_metrics(summary_stats["MC-Dropout (Scratch Model #0)"])
    ens_cnt, ens_m_exp, ens_med_exp, ens_p90_exp, ens_m_cost, ens_m_time, ens_m_lam = calc_metrics(summary_stats["Calibrated Conf-Aware Ensemble (Ours)"])

    report_content = fr"""# Benchmark Report: 5-Box OOD Calibrated Experiments
### MC-Dropout on Scratch Model #0 vs. Calibrated Epistemic Ensemble Gating

---

## 1. Experimental Overview
* **Dataset:** 5-Box Sokoban (100 Out-of-Distribution Mazes)
* **Compute Platform:** {device}
* **Search Timeout:** {args.timeout}s (6.0 minutes hard ceiling)
* **Expansion Limit:** {args.max_exp} (Uncapped time-priority search)
* **Calibrated Gating Hyperparameters:** Standard Deviation Mode, $\\beta = {args.beta}$, $\\lambda_{{\\min}} = {args.lambda_min}$
* **Total Runtime:** {total_benchmark_time / 60.0:.2f} minutes

---

## 2. Benchmark Summary Table

| Algorithm | Solve Rate | Plan Verified | Mean Expansions | Median Expansions | 90th % Expansions | Mean Cost | Mean Time (ms) | Mean $\\bar{{\\lambda}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MC-Dropout (Scratch Model #0)** | {mc_cnt / n_instances * 100:.1f}% ({mc_cnt}/{n_instances}) | 100.0% | {mc_m_exp:.1f} | {mc_med_exp:.1f} | {mc_p90_exp:.1f} | {mc_m_cost:.1f} | {mc_m_time:.1f} ms | {mc_m_lam:.3f} |
| **Calibrated Conf-Aware Ensemble (Ours)** | {ens_cnt / n_instances * 100:.1f}% ({ens_cnt}/{n_instances}) | 100.0% | **{ens_m_exp:.1f}** | **{ens_med_exp:.1f}** | **{ens_p90_exp:.1f}** | {ens_m_cost:.1f} | {ens_m_time:.1f} ms | {ens_m_lam:.3f} |

---

## 3. Comparative Context vs. Reference Baselines
* **Paper Learned A\* (Deterministic)**: 71.0% Solved | Mean Exp: 2,833.8 | Median Exp: 1,384.0
* **MC-Dropout on Paper Model**: 90.0% Solved | Mean Exp: 1,970.5 | Median Exp: 567.5
* **Scratch Model #0 (Deterministic)**: 93.0% Solved | Mean Exp: 8,740.4 | Median Exp: 1,436.0
* **Normal 5-Model Ensemble (Pure Mean)**: 95.0% Solved | Mean Exp: 1,665.1 | Median Exp: 591.0
"""
    os.makedirs(os.path.dirname(args.output_report) if os.path.dirname(args.output_report) else ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "=" * 105, flush=True)
    print(" 5-BOX CALIBRATED BENCHMARK COMPLETE!", flush=True)
    print(f" Summary Results saved to: {args.output_csv}", flush=True)
    print(f" Comprehensive Report   : {args.output_report}", flush=True)
    print(f" Total Runtime          : {total_benchmark_time / 60.0:.2f} minutes", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    main()
