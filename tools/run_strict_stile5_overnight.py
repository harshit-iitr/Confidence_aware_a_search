import os
import sys
import csv
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.torch_model import ChrestienHeuristicNet
from src.sliding_tile.env import load_sliding_tile_boards, manhattan_linear_conflict
from src.sliding_tile.astar_sliding_tile import solve_sliding_tile_astar_strict


def update_live_markdown_report(csv_path: str, report_path: str, methods: list):
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    # Find boards completed by all methods for fair head-to-head comparison
    by_board = {}
    for r in rows:
        b_idx = int(r["board_idx"])
        by_board.setdefault(b_idx, {})[r["method"]] = r

    common_boards = sorted(b for b, m_dict in by_board.items() if all(m in m_dict for m in methods))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Sliding-Tile ($5\\times 5$ 24-Puzzle) Strict Paper A* Benchmark (No Expansion Limit, 6-Min Timeout)\n\n")
        f.write(f"- **Model:** Pre-trained `stile` checkpoint (`checkpoints/slidingtile/stile_paper_pytorch.pt`)\n")
        f.write(f"- **Search Settings:** Strict `test_on_Astar.py` implementation (`max_expansions = None`, `timeout = 360.0s` / 6 min per board)\n")
        f.write(f"- **Completed Boards So Far:** {len(common_boards)} / 200 boards from `test/stile5.txt`\n\n")

        f.write("## 1. Overall Results Across Completed Boards\n\n")
        f.write("| Method | Solved (%) | Mean Paper Expansions (Solved) | Median Paper Expansions (Solved) | Mean Nodes Popped (Solved) | Mean Plan Length | Mean Runtime (s) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for m in methods:
            m_rows = [by_board[b][m] for b in common_boards]
            if not m_rows:
                continue
            solved_rows = [r for r in m_rows if int(r["solved"]) == 1]
            sr = 100.0 * len(solved_rows) / len(m_rows)
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

        # Head-to-head on mutually solved boards
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
            f.write(f"\n## 2. Head-to-Head on {len(both_ra)} Common Solved Boards (`mc_dropout_risk_averse` vs `paper_baseline_lstar`)\n\n")
            f.write(f"- **Paper Baseline Mean / Median Expansions:** {np.mean(base_exps):.1f} / {np.median(base_exps):.1f}\n")
            f.write(f"- **MC-Dropout Risk-Averse Mean / Median Expansions:** {np.mean(ra_exps):.1f} / {np.median(ra_exps):.1f}\n")
            f.write(f"- **Win / Tie / Loss Record:** {wins}W / {ties}T / {losses}L\n")


def main():
    device = torch.device(
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"=== Strict Paper A* Sliding-Tile Benchmark (No Limit, 6-Min Timeout) on {device} ===", flush=True)

    ckpt_path = "checkpoints/slidingtile/stile_paper_pytorch.pt"
    model = ChrestienHeuristicNet(dim=5, in_channels=50).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded pre-trained paper model from {ckpt_path}", flush=True)

    test_path = "Optimize-Planning-Heuristics-to-Rank/slidingtile/test/stile5.txt"
    boards = load_sliding_tile_boards(test_path, dim=5, fix_parity=False)
    print(f"Loaded {len(boards)} test boards from {test_path}", flush=True)

    methods = [
        "paper_baseline_lstar",
        "mc_dropout_risk_averse",
        "mc_dropout_conf_aware",
    ]

    csv_path = "results/sliding_tile_strict_paper_6min.csv"
    report_path = "results/sliding_tile_strict_paper_6min_report.md"
    os.makedirs("results", exist_ok=True)

    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                completed.add((int(row["board_idx"]), row["method"]))

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    f_out = open(csv_path, "a", newline="")
    writer = csv.writer(f_out)
    if write_header:
        writer.writerow([
            "board_idx", "init_md_lc", "method", "solved",
            "paper_expansions", "nodes_popped", "plan_length",
            "runtime", "mean_sigma", "mean_lambda"
        ])
        f_out.flush()

    for b_idx, board in enumerate(boards):
        init_md = manhattan_linear_conflict(board, 5)
        for m_name in methods:
            if (b_idx, m_name) in completed:
                continue

            t_b = time.time()
            res = solve_sliding_tile_astar_strict(
                init_board=board,
                dim=5,
                mode=m_name,
                model=model,
                device=device,
                timeout=360.0,
                max_expansions=None,
                kappa=0.50,
                beta=2.5,
                lambda_min=0.75,
                mc_samples=10,
                mc_dropout_p=0.20,
            )

            writer.writerow([
                b_idx, f"{init_md:.1f}", m_name, int(res["solved"]),
                res["paper_expansions"], res["nodes_popped"], res["plan_length"],
                f"{res['runtime']:.4f}", f"{res['mean_sigma']:.4f}", f"{res['mean_lambda']:.4f}"
            ])
            f_out.flush()

            status = "SOLVED" if res["solved"] else "TIMEOUT"
            print(
                f"[Board {b_idx:3d}/200 | MD+LC={init_md:4.0f}] {m_name:24s} -> {status:7s} | "
                f"paper_exp={res['paper_expansions']:6d} | popped={res['nodes_popped']:6d} | "
                f"len={res['plan_length']:3d} | time={res['runtime']:6.2f}s | sigma={res['mean_sigma']:.2f}",
                flush=True,
            )

        update_live_markdown_report(csv_path, report_path, methods)

    f_out.close()
    print("\n=== Strict Paper A* Sliding-Tile Benchmark Complete! ===", flush=True)


if __name__ == "__main__":
    main()
