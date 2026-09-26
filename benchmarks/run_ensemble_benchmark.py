import os
import sys
import time
import argparse
import csv
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.torch_model import ChrestienHeuristicNet
from src.sokoban_env import SokobanEnv
from src.classical_heuristics import manhattan_distance_heuristic
from src.confidence_aware_search import get_device, author_state_to_tensor
from src.ensemble_search import DeepEnsembleHeuristic, run_ensemble_astar
from benchmarks.run_5box_benchmark import verify_solution_plan, run_batched_paper_search


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Deep Ensemble on Sokoban Benchmark")
    parser.add_argument("--dataset", type=str, default="data/sokoban_5box_test.txt", help="Path to test dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ensemble", help="Directory with ensemble_model_*.pt")
    parser.add_argument("--num_mazes", type=int, default=100, help="Number of instances to evaluate")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-algorithm timeout in seconds")
    parser.add_argument("--max_exp", type=int, default=15000, help="Max node expansions")
    parser.add_argument("--output_csv", type=str, default="results/ensemble_benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--output_report", type=str, default="results/ensemble_benchmark_report.md", help="Output report path")
    args = parser.parse_args()

    device = get_device()
    print("=" * 80, flush=True)
    print(" DEEP ENSEMBLE CONFIDENCE-AWARE HEURISTIC BENCHMARK", flush=True)
    print(f" Compute Device  : {device}", flush=True)
    print(f" Test Dataset    : {args.dataset}", flush=True)
    print(f" Checkpoint Dir  : {args.checkpoint_dir}", flush=True)
    print(f" Max Instances   : {args.num_mazes}", flush=True)
    print(f" Timeout         : {args.timeout}s", flush=True)
    print(f" Expansion Limit : {args.max_exp}", flush=True)
    print("=" * 80, flush=True)

    # Find ensemble checkpoints
    model_paths = []
    if os.path.exists(args.checkpoint_dir):
        for f in sorted(os.listdir(args.checkpoint_dir)):
            if f.endswith(".pt") and "ensemble_model_" in f:
                model_paths.append(os.path.join(args.checkpoint_dir, f))

    if not model_paths:
        raise FileNotFoundError(f"No ensemble checkpoints ('ensemble_model_*.pt') found in '{args.checkpoint_dir}'! Please train models first using tools/train_deep_ensemble.py.")

    print(f"Found {len(model_paths)} trained ensemble checkpoints:", flush=True)
    for p in model_paths:
        print(f"  - {p}", flush=True)

    models = []
    for p in model_paths:
        m = ChrestienHeuristicNet(dim=10).to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)

    all_states = SokobanEnv.load_dataset(args.dataset)
    test_states = all_states[:args.num_mazes]
    n_instances = len(test_states)
    print(f"\nLoaded {n_instances} benchmark instances for evaluation.\n", flush=True)

    os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else ".", exist_ok=True)
    csv_file = open(args.output_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "map_id", "algorithm", "solved", "expansions", "cost", "time_sec",
        "mean_lambda", "mean_variance", "plan_length", "verified"
    ])

    summary_data = {
        "Deterministic Single Model (Baseline)": {"sol": 0, "exp": [], "cost": [], "time": [], "verified": 0},
        "Deep Ensemble Confidence-Aware A* (Ours)": {"sol": 0, "exp": [], "cost": [], "time": [], "verified": 0, "lambdas": [], "vars": []}
    }

    start_total = time.time()
    for map_idx, state in enumerate(test_states, 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)
        goal_t = torch.from_numpy(author_state_to_tensor(goal_state, box_targets, 10)).unsqueeze(0).to(device)

        print(f"Evaluating Map {map_idx:03d} / {n_instances} ...", flush=True)

        # 1. Deterministic baseline (Model #0)
        b_sol, b_exp, b_cost, b_plan, b_time = run_batched_paper_search(
            state, box_targets, models[0], goal_t, device, alg="astar", max_time=args.timeout, max_expansions=args.max_exp, dim=10
        )
        b_ver = verify_solution_plan(state, b_plan, box_targets, 10) if b_sol else False
        if b_sol and b_ver:
            summary_data["Deterministic Single Model (Baseline)"]["sol"] += 1
            summary_data["Deterministic Single Model (Baseline)"]["verified"] += 1
            summary_data["Deterministic Single Model (Baseline)"]["exp"].append(b_exp)
            summary_data["Deterministic Single Model (Baseline)"]["cost"].append(b_cost)
            summary_data["Deterministic Single Model (Baseline)"]["time"].append(b_time)
        csv_writer.writerow([map_idx, "Deterministic Single Model (Baseline)", b_sol, b_exp, b_cost, f"{b_time:.4f}", "", "", len(b_plan), b_ver])
        print(f"  [1] Single Model Baseline  : {'SOLVED ('+str(b_exp)+' exp)' if b_sol else 'TIMEOUT'} in {b_time*1000:6.1f}ms", flush=True)

        # 2. Deep Ensemble Confidence-Aware A*
        ensemble_heur = DeepEnsembleHeuristic(models, goal_state, box_targets, device, dim=10)
        e_sol, e_exp, e_cost, e_plan, e_time, e_stats = run_ensemble_astar(
            state, box_targets, ensemble_heur, max_expansions=args.max_exp, max_time=args.timeout, dim=10
        )
        e_ver = verify_solution_plan(state, e_plan, box_targets, 10) if e_sol else False
        if e_sol and e_ver:
            summary_data["Deep Ensemble Confidence-Aware A* (Ours)"]["sol"] += 1
            summary_data["Deep Ensemble Confidence-Aware A* (Ours)"]["verified"] += 1
            summary_data["Deep Ensemble Confidence-Aware A* (Ours)"]["exp"].append(e_exp)
            summary_data["Deep Ensemble Confidence-Aware A* (Ours)"]["cost"].append(e_cost)
            summary_data["Deep Ensemble Confidence-Aware A* (Ours)"]["time"].append(e_time)
            summary_data["Deep Ensemble Confidence-Aware A* (Ours)"]["lambdas"].append(e_stats["mean_lambda"])
            summary_data["Deep Ensemble Confidence-Aware A* (Ours)"]["vars"].append(e_stats["mean_variance"])
        csv_writer.writerow([
            map_idx, "Deep Ensemble Confidence-Aware A* (Ours)", e_sol, e_exp, e_cost, f"{e_time:.4f}",
            f"{e_stats['mean_lambda']:.4f}", f"{e_stats['mean_variance']:.4f}", len(e_plan), e_ver
        ])
        csv_file.flush()
        e_lbl = f"lambda={e_stats['mean_lambda']:.2f}" if e_sol else "timed out"
        print(f"  [2] Deep Ensemble A* (Ours): {'SOLVED ('+str(e_exp)+' exp, '+e_lbl+')' if e_sol else 'TIMEOUT'} in {e_time*1000:6.1f}ms", flush=True)

    csv_file.close()
    elapsed_total = time.time() - start_total

    print("\n" + "=" * 80, flush=True)
    print(f" FINAL ENSEMBLE REPORT ({n_instances} Instances, Elapsed: {elapsed_total/60:.2f}m)", flush=True)
    print("=" * 80, flush=True)
    for alg, d in summary_data.items():
        sol = d["sol"]
        sol_pct = (sol / n_instances) * 100.0 if n_instances > 0 else 0.0
        m_exp = f"{np.mean(d['exp']):.1f}" if sol > 0 else "N/A"
        med_exp = f"{np.median(d['exp']):.1f}" if sol > 0 else "N/A"
        print(f"{alg:<42} | Solved: {sol}/{n_instances} ({sol_pct:.1f}%) | Mean Exp: {m_exp:<8} | Med Exp: {med_exp:<8}", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
