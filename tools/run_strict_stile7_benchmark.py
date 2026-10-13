"""
================================================================================
Sliding-Tile (7x7 / 48-Puzzle) Strict Paper A* Benchmark Suite
================================================================================
Evaluates the pre-trained paper heuristic network (stile_paper_pytorch.pt) on
7x7 (48-Puzzle) test boards from stile7.txt across:
  1. Paper Baseline A* (L* ranking heuristic)
  2. MC-Dropout Risk-Averse A* (kappa = 0.50)
  3. MC-Dropout Confidence-Aware A* (beta = 2.5)

Features:
  - Strict adherence to paper evaluation (test_on_Astar.py): records both
    paper_expansions and nodes_popped.
  - Resume safe: skips already evaluated (board_idx, method) pairs in CSV.
  - Live markdown report generation after every completed board.
  - Hardware agnostic: auto-detects CUDA GPU, Intel Arc XPU, or CPU fallback.
================================================================================
"""

import os
import sys
import csv
import time
import argparse
from typing import List, Dict, Optional

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.torch_model import ChrestienHeuristicNet
from src.sliding_tile.env import load_sliding_tile_boards, manhattan_linear_conflict
from src.sliding_tile.astar_sliding_tile import solve_sliding_tile_astar_strict


def update_live_markdown_report(csv_path: str, report_path: str, methods: List[str], dim: int = 7):
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    by_board: Dict[int, Dict[str, dict]] = {}
    for r in rows:
        b_idx = int(r["board_idx"])
        by_board.setdefault(b_idx, {})[r["method"]] = r

    common_boards = sorted(b for b, m_dict in by_board.items() if all(m in m_dict for m in methods))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Sliding-Tile ({dim}x{dim} 48-Puzzle) Strict Paper A* Benchmark Report\n\n")
        f.write(f"- **Grid Dimension:** {dim}x{dim} (48-Puzzle)\n")
        f.write(f"- **Model:** Pre-trained Paper Checkpoint (`checkpoints/slidingtile/stile_paper_pytorch.pt`)\n")
        f.write(f"- **Search Settings:** Strict `test_on_Astar.py` (No artificial expansion limit, 360s / 6-min timeout)\n")
        f.write(f"- **Completed Boards (All Methods):** {len(common_boards)} boards evaluated\n")
        f.write(f"- **Total Rows Logged:** {len(rows)}\n\n")

        f.write("## 1. Overall Results Across Completed Boards\n\n")
        f.write("| Method | Solved (%) | Mean Paper Expansions | Median Paper Expansions | Mean Nodes Popped | Mean Plan Length | Mean Runtime (s) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for m in methods:
            m_rows = [by_board[b][m] for b in common_boards if m in by_board[b]]
            if not m_rows:
                continue
            solved_rows = [r for r in m_rows if int(r["solved"]) == 1]
            sr = 100.0 * len(solved_rows) / max(1, len(m_rows))
            sr_str = f"{sr:.1f}% ({len(solved_rows)}/{len(m_rows)})"

            if solved_rows:
                p_exp_mean = f"{np.mean([int(r['paper_expansions']) for r in solved_rows]):.1f}"
                p_exp_med = f"{np.median([int(r['paper_expansions']) for r in solved_rows]):.1f}"
                pop_mean = f"{np.mean([int(r['nodes_popped']) for r in solved_rows]):.1f}"
                plen_mean = f"{np.mean([int(r['plan_length']) for r in solved_rows]):.1f}"
            else:
                p_exp_mean = p_exp_med = pop_mean = plen_mean = "N/A"
            rt_mean = f"{np.mean([float(r['runtime']) for r in m_rows]):.2f}s"

            f.write(f"| `{m}` | **{sr_str}** | {p_exp_mean} | {p_exp_med} | {pop_mean} | {plen_mean} | {rt_mean} |\n")

        # Head-to-Head Comparisons
        if "paper_baseline_lstar" in methods and "mc_dropout_risk_averse" in methods:
            both_ra = [
                b for b in common_boards
                if int(by_board[b]["paper_baseline_lstar"]["solved"]) == 1
                and int(by_board[b]["mc_dropout_risk_averse"]["solved"]) == 1
            ]
            if both_ra:
                base_exps = np.array([int(by_board[b]["paper_baseline_lstar"]["paper_expansions"]) for b in both_ra])
                ra_exps = np.array([int(by_board[b]["mc_dropout_risk_averse"]["paper_expansions"]) for b in both_ra])
                wins = int(np.sum(ra_exps < base_exps))
                ties = int(np.sum(ra_exps == base_exps))
                losses = int(np.sum(ra_exps > base_exps))
                f.write(f"\n## 2. Head-to-Head: `mc_dropout_risk_averse` vs `paper_baseline_lstar` ({len(both_ra)} boards)\n\n")
                f.write(f"- **Paper Baseline Mean / Median:** {np.mean(base_exps):.1f} / {np.median(base_exps):.1f}\n")
                f.write(f"- **Risk-Averse Mean / Median:** {np.mean(ra_exps):.1f} / {np.median(ra_exps):.1f}\n")
                f.write(f"- **Win / Tie / Loss Record:** **{wins}W / {ties}T / {losses}L**\n")

        if "paper_baseline_lstar" in methods and "mc_dropout_conf_aware" in methods:
            both_ca = [
                b for b in common_boards
                if int(by_board[b]["paper_baseline_lstar"]["solved"]) == 1
                and int(by_board[b]["mc_dropout_conf_aware"]["solved"]) == 1
            ]
            if both_ca:
                base_exps = np.array([int(by_board[b]["paper_baseline_lstar"]["paper_expansions"]) for b in both_ca])
                ca_exps = np.array([int(by_board[b]["mc_dropout_conf_aware"]["paper_expansions"]) for b in both_ca])
                wins = int(np.sum(ca_exps < base_exps))
                ties = int(np.sum(ca_exps == base_exps))
                losses = int(np.sum(ca_exps > base_exps))
                f.write(f"\n## 3. Head-to-Head: `mc_dropout_conf_aware` vs `paper_baseline_lstar` ({len(both_ca)} boards)\n\n")
                f.write(f"- **Paper Baseline Mean / Median:** {np.mean(base_exps):.1f} / {np.median(base_exps):.1f}\n")
                f.write(f"- **Confidence-Aware Mean / Median:** {np.mean(ca_exps):.1f} / {np.median(ca_exps):.1f}\n")
                f.write(f"- **Win / Tie / Loss Record:** **{wins}W / {ties}T / {losses}L**\n")


def run_stile7_benchmark():
    parser = argparse.ArgumentParser(description="Sliding-Tile (7x7 48-Puzzle) Strict Paper Benchmark")
    parser.add_argument("--test_file", type=str, default="Optimize-Planning-Heuristics-to-Rank/slidingtile/test/stile7.txt")
    parser.add_argument("--ckpt", type=str, default="checkpoints/slidingtile/stile_paper_pytorch.pt")
    parser.add_argument("--csv_output", type=str, default="results/sliding_tile_7x7_strict_paper_benchmark.csv")
    parser.add_argument("--report_output", type=str, default="results/sliding_tile_7x7_strict_paper_report.md")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-board timeout in seconds (default: 180s = 3 mins)")
    parser.add_argument("--max_boards", type=int, default=200, help="Number of boards to benchmark")
    parser.add_argument("--start_idx", type=int, default=0, help="Start board index")
    parser.add_argument("--end_idx", type=int, default=None, help="End board index")
    parser.add_argument("--max_expansions", type=int, default=None, help="Max expansions cap (default None = strict paper run)")
    parser.add_argument("--methods", type=str, default="paper_baseline_lstar,mc_dropout_risk_averse,mc_dropout_conf_aware")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'cuda', 'xpu', or 'cpu'")
    parser.add_argument("--kappa", type=float, default=0.50, help="Risk-averse penalty coefficient")
    parser.add_argument("--beta", type=float, default=2.5, help="Confidence decay rate")
    args = parser.parse_args()

    # Hardware device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available()
            else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
        )

    print("====================================================================", flush=True)
    print("Sliding-Tile (7x7 / 48-Puzzle) Strict Paper A* Benchmark", flush=True)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB VRAM)", flush=True)
    elif device.type == "xpu":
        print(f"GPU: {torch.xpu.get_device_name(0)}", flush=True)
    print("====================================================================", flush=True)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Model checkpoint not found at: {args.ckpt}")
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test boards file not found at: {args.test_file}")

    print(f"Loading paper model checkpoint from {args.ckpt}...", flush=True)
    model = ChrestienHeuristicNet(dim=5, in_channels=50).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    print(f"Loading 7x7 test boards from {args.test_file}...", flush=True)
    all_boards = load_sliding_tile_boards(args.test_file, dim=7, fix_parity=True)
    start = args.start_idx
    end = min(len(all_boards), args.end_idx if args.end_idx is not None else start + args.max_boards)
    boards_to_test = list(enumerate(all_boards[start:end], start=start))
    print(f"Total boards in range [{start}, {end}): {len(boards_to_test)}", flush=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"Methods to evaluate: {methods}", flush=True)

    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)
    completed = set()
    if os.path.exists(args.csv_output):
        with open(args.csv_output, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                completed.add((int(row["board_idx"]), row["method"]))
        print(f"Found {len(completed)} existing results in {args.csv_output}. Resuming...", flush=True)

    write_header = not os.path.exists(args.csv_output) or os.path.getsize(args.csv_output) == 0
    f_csv = open(args.csv_output, "a", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    if write_header:
        writer.writerow([
            "board_idx", "dim", "init_md_lc", "method", "solved",
            "paper_expansions", "nodes_popped", "plan_length",
            "runtime", "mean_sigma", "mean_lambda"
        ])
        f_csv.flush()

    for idx, board in boards_to_test:
        init_md = manhattan_linear_conflict(board, 7)
        print(f"\n--- [Board #{idx:3d}/{len(all_boards)} | Init MD+LC: {init_md:.1f}] ---", flush=True)
        for m_name in methods:
            if (idx, m_name) in completed:
                print(f"  [{m_name}] Already evaluated. Skipping.", flush=True)
                continue

            t_start = time.time()
            res = solve_sliding_tile_astar_strict(
                init_board=board,
                dim=7,
                mode=m_name,
                model=model,
                device=device,
                timeout=args.timeout,
                max_expansions=args.max_expansions,
                kappa=args.kappa,
                beta=args.beta,
            )
            elapsed = time.time() - t_start

            writer.writerow([
                idx, 7, f"{init_md:.1f}", m_name, int(res["solved"]),
                res["paper_expansions"], res["nodes_popped"], res["plan_length"],
                f"{res['runtime']:.4f}", f"{res['mean_sigma']:.4f}", f"{res['mean_lambda']:.4f}"
            ])
            f_csv.flush()
            completed.add((idx, m_name))

            status_str = f"SOLVED ({res['paper_expansions']} exp, len {res['plan_length']})" if res["solved"] else f"TIMEOUT ({res['paper_expansions']} exp)"
            print(f"  [{m_name}] {status_str} in {elapsed:.2f}s", flush=True)

        update_live_markdown_report(args.csv_output, args.report_output, methods, dim=7)

    f_csv.close()
    print("\n====================================================================", flush=True)
    print(f"Benchmark completed! Final CSV saved to: {args.csv_output}", flush=True)
    print(f"Markdown report updated at: {args.report_output}", flush=True)
    print("====================================================================", flush=True)


if __name__ == "__main__":
    run_stile7_benchmark()
