import os
import sys
import time
import argparse
import csv
from typing import List, Dict, Optional
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.torch_model import ChrestienHeuristicNet
from src.sokoban_env import SokobanEnv
from src.confidence_aware_search import get_device, get_device_name
from src.ensemble_search import DeepEnsembleHeuristic, run_ensemble_astar

DEFAULT_HARD_MAPS = [2, 10, 12, 14, 22, 26, 27, 34, 42, 62]


def load_reference_results() -> Dict[int, Dict[str, Dict]]:
    refs = {}
    files = [
        ("results/ensemble_5box_benchmark_results.csv", "XPU"),
        ("results/riskaverse_5box_benchmark_results.csv", "CUDA")
    ]
    for path, tag in files:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        m_id = int(r["map_id"])
                        if m_id not in refs:
                            refs[m_id] = {}
                        key = f"{r['algorithm'].strip()} [{tag}]"
                        refs[m_id][key] = {
                            "solved": r["solved"].strip().lower() == "true",
                            "expansions": int(float(r["expansions"])),
                            "cost": float(r["path_cost"]) if r["solved"].strip().lower() == "true" else float("inf"),
                            "time_sec": float(r["time_sec"])
                        }
                    except (ValueError, KeyError):
                        continue
    return refs


def fmt_ref(entry: Optional[Dict]) -> str:
    if not entry:
        return "N/A"
    if not entry["solved"]:
        return f"TIMEOUT ({entry['expansions']} exp)"
    return f"{entry['expansions']} exp ({entry['time_sec']:.1f}s, cost={entry['cost']:.0f})"


def main():
    parser = argparse.ArgumentParser(description="10 Hard 5-Box Maps: Goal-Anchored Risk-Averse vs. Earlier Percentile Risk-Averse on XPU")
    parser.add_argument("--dataset", type=str, default="data/sokoban_5box_test.txt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ensemble")
    parser.add_argument("--maps", type=int, nargs="+", default=DEFAULT_HARD_MAPS, help="1-indexed Map IDs to evaluate")
    parser.add_argument("--timeout", type=float, default=360.0, help="Per-map timeout in seconds (6.0 minutes)")
    parser.add_argument("--max_exp", type=int, default=1000000)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--goal_scale", type=float, default=1.5, help="Scale factor for Goal-Anchored 5-box metric heuristic")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_csv", type=str, default="results/hard10_goal_anchored_vs_riskaverse.csv")
    parser.add_argument("--output_report", type=str, default="results/hard10_goal_anchored_vs_riskaverse_report.md")
    args = parser.parse_args()

    device = get_device(args.device)
    dev_name = get_device_name(device)

    print("=" * 108, flush=True)
    print(" 10 HARD 5-BOX MAPS BENCHMARK: GOAL-ANCHORED RISK-AVERSE vs. EARLIER PERCENTILE RISK-AVERSE", flush=True)
    print(f" Compute Device       : {device} [{dev_name}]", flush=True)
    print(f" Maps Selected (10)   : {args.maps}", flush=True)
    print(f" Per-Map Timeout      : {args.timeout}s (6.0 minutes hard limit per map)", flush=True)
    print(f" Risk-Averse Kappa    : {args.kappa}", flush=True)
    print(f" Goal-Anchored Scale  : {args.goal_scale}", flush=True)
    print("=" * 108, flush=True)

    models = []
    for i in range(5):
        p = os.path.join(args.checkpoint_dir, f"ensemble_model_{i}.pt")
        m = ChrestienHeuristicNet(dim=10).to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)
    print(f"Loaded all {len(models)} ensemble models to {device}.\n", flush=True)

    all_mazes = SokobanEnv.load_dataset(args.dataset)
    refs = load_reference_results()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    csv_file = open(args.output_csv, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "map_id", "algorithm", "solved", "expansions", "path_cost",
        "time_sec", "mean_variance", "plan_length", "plan_verified"
    ])
    csv_file.flush()

    alg_pct = f"Earlier Risk-Averse (Percentile, kappa={args.kappa})"
    alg_ga = f"Goal-Anchored Risk-Averse (h(s,g)-h(g,g), kappa={args.kappa}, scale={args.goal_scale})"
    algs = [alg_pct, alg_ga]

    results_by_map = {m_id: {} for m_id in args.maps}
    t_start_all = time.time()

    for idx, m_id in enumerate(args.maps, 1):
        state = all_mazes[m_id - 1]
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)

        ref_xpu_mean = refs.get(m_id, {}).get("Normal 5-Model Ensemble A* (Pure Mean) [XPU]")
        ref_cuda_ra = refs.get(m_id, {}).get("Risk-Averse Deep Ensemble A* (kappa=1.0) [CUDA]")

        print("-" * 108, flush=True)
        print(
            f" [{idx:02d}/{len(args.maps):02d}] Map {m_id:03d} | "
            f"Ref XPU PureMean: {fmt_ref(ref_xpu_mean)} | Ref CUDA RiskAverse: {fmt_ref(ref_cuda_ra)}",
            flush=True
        )

        # 1. Earlier Percentile Risk-Averse
        heur_pct = DeepEnsembleHeuristic(
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
        sol1, exp1, cost1, plan1, dt1, stats1 = run_ensemble_astar(
            init_state=state,
            box_targets=box_targets,
            ensemble_heuristic=heur_pct,
            max_expansions=args.max_exp,
            max_time=args.timeout,
            dim=10,
            fixed_lambda=None,
            prune_deadlocks=False
        )
        ver1 = SokobanEnv.verify_solution_plan(state, plan1, box_targets, 10) if sol1 else False
        writer.writerow([
            m_id, alg_pct, sol1, exp1, cost1 if sol1 else "inf",
            round(dt1, 4), round(stats1.get("mean_variance", 0.0), 4), len(plan1), ver1
        ])
        csv_file.flush()
        results_by_map[m_id][alg_pct] = {"solved": sol1 and ver1, "exp": exp1, "cost": cost1, "time": dt1}
        s1_str = f"{exp1} exp ({dt1:.2f}s) | Cost: {cost1:.0f} | Verified [OK]" if (sol1 and ver1) else f"FAILED (Timeout @ {exp1} exp, {dt1:.1f}s)"
        print(f"  -> {alg_pct:<68}: {s1_str}", flush=True)

        # 2. Goal-Anchored Risk-Averse
        heur_ga = DeepEnsembleHeuristic(
            models=models,
            goal_state=goal_state,
            box_targets=box_targets,
            device=device,
            search_mode="goal_anchored",
            fallback_type="manhattan",
            kappa=args.kappa,
            scale_constant_C=args.goal_scale,
            dim=10
        )
        sol2, exp2, cost2, plan2, dt2, stats2 = run_ensemble_astar(
            init_state=state,
            box_targets=box_targets,
            ensemble_heuristic=heur_ga,
            max_expansions=args.max_exp,
            max_time=args.timeout,
            dim=10,
            fixed_lambda=None,
            prune_deadlocks=False
        )
        ver2 = SokobanEnv.verify_solution_plan(state, plan2, box_targets, 10) if sol2 else False
        writer.writerow([
            m_id, alg_ga, sol2, exp2, cost2 if sol2 else "inf",
            round(dt2, 4), round(stats2.get("mean_variance", 0.0), 4), len(plan2), ver2
        ])
        csv_file.flush()
        results_by_map[m_id][alg_ga] = {"solved": sol2 and ver2, "exp": exp2, "cost": cost2, "time": dt2}
        s2_str = f"{exp2} exp ({dt2:.2f}s) | Cost: {cost2:.0f} | Verified [OK] | step_var={stats2.get('mean_variance',0.0):.2f}" if (sol2 and ver2) else f"FAILED (Timeout @ {exp2} exp, {dt2:.1f}s)"
        print(f"  -> {alg_ga:<68}: {s2_str}", flush=True)

    csv_file.close()
    total_min = (time.time() - t_start_all) / 60.0
    print("\n" + "=" * 108, flush=True)
    print(f" COMPLETED 10 HARD MAPS IN {total_min:.2f} MINUTES", flush=True)
    print("=" * 108, flush=True)

    # Generate markdown report
    lines = [
        "# 10 Hard 5-Box Maps: Goal-Anchored vs. Earlier Percentile Risk-Averse Search",
        f"* **Compute Device:** `{device} [{dev_name}]`",
        f"* **Per-Map Timeout:** `{args.timeout}s` (6.0 minutes)",
        f"* **Total Runtime:** `{total_min:.2f} minutes`",
        "",
        "| Map ID | XPU Pure Mean (Ref) | Earlier Risk-Averse (Percentile, XPU) | Goal-Anchored Risk-Averse (XPU) |",
        "| :---: | :---: | :---: | :---: |"
    ]
    for m_id in args.maps:
        ref_xpu = fmt_ref(refs.get(m_id, {}).get("Normal 5-Model Ensemble A* (Pure Mean) [XPU]"))
        r1 = results_by_map[m_id][alg_pct]
        r2 = results_by_map[m_id][alg_ga]
        c1 = f"**{r1['exp']} exp** ({r1['time']:.1f}s, cost={r1['cost']:.0f})" if r1["solved"] else f"TIMEOUT ({r1['exp']} exp)"
        c2 = f"**{r2['exp']} exp** ({r2['time']:.1f}s, cost={r2['cost']:.0f})" if r2["solved"] else f"TIMEOUT ({r2['exp']} exp)"
        lines.append(f"| **Map {m_id}** | {ref_xpu} | {c1} | {c2} |")

    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved report to {args.output_report}", flush=True)


if __name__ == "__main__":
    main()
