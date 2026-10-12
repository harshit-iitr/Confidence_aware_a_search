import os
import sys
import csv
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.torch_model import ChrestienHeuristicNet
from src.sliding_tile.env import (
    get_goal_board,
    load_sliding_tile_boards,
    manhattan_linear_conflict,
    encode_boards_25ch,
)
from src.sliding_tile.astar_sliding_tile import solve_sliding_tile_astar
from src.sliding_tile.train_ensemble import train_sliding_tile_ensemble


def generate_ood_benchmark_boards(dim: int, count: int = 40, min_steps: int = 24, max_steps: int = 46) -> list:
    """
    Generates a deterministic, reproducible suite of solvable OOD Sliding-Tile boards
    for 6x6 (35-Puzzle) and 7x7 (48-Puzzle) across increasing scramble complexities.
    """
    cache_dir = "data/slidingtile"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"ood_{dim}x{dim}_benchmark.txt")

    if os.path.exists(cache_file):
        return load_sliding_tile_boards(cache_file, dim, fix_parity=True)[:count]

    goal = get_goal_board(dim)
    boards = []
    seen = {goal.tobytes()}

    for i in range(count):
        rng = np.random.default_rng(5000 + dim * 1000 + i)
        steps = int(min_steps + (max_steps - min_steps) * (i / max(1, count - 1)))
        cur = goal.copy()
        pr, pc = -1, -1
        for _ in range(steps):
            br, bc = np.argwhere(cur == 0)[0]
            cands = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = br + dr, bc + dc
                if 0 <= nr < dim and 0 <= nc < dim and (nr, nc) != (pr, pc):
                    nb = cur.copy()
                    nb[br, bc], nb[nr, nc] = nb[nr, nc], 0
                    cands.append((manhattan_linear_conflict(nb, dim), nb, nr, nc))
            cands.sort(key=lambda x: x[0], reverse=True)
            idx = 0 if (rng.random() < 0.65 or len(cands) == 1) else int(rng.integers(1, len(cands)))
            pr, pc = br, bc
            cur = cands[idx][1]
        boards.append(cur)
        seen.add(cur.tobytes())

    with open(cache_file, "w") as f:
        for b in boards:
            f.write(" ".join(str(int(x)) for x in b.ravel()) + "\n")
    return boards


def run_benchmark_suite(
    suite_name: str,
    boards: list,
    dim: int,
    paper_model: ChrestienHeuristicNet,
    ensemble_models: list,
    device: torch.device,
    csv_path: str,
    max_expansions: int = 2000,
    timeout: float = 14.0,
):
    methods = [
        ("classical_md_lc", []),
        ("paper_baseline_lstar", [paper_model]),
        ("mc_dropout_conf_aware", [paper_model]),
        ("mc_dropout_risk_averse", [paper_model]),
        ("single_scratch_m0", [ensemble_models[0]]),
        ("normal_ensemble", ensemble_models),
        ("conf_aware_ensemble", ensemble_models),
        ("risk_averse_ensemble", ensemble_models),
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["suite"], int(row["board_idx"]), row["method"]))

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    f_out = open(csv_path, "a", newline="")
    writer = csv.writer(f_out)
    if write_header:
        writer.writerow([
            "suite", "dim", "board_idx", "init_md_lc", "method",
            "solved", "expansions", "plan_length", "runtime", "mean_sigma", "mean_lambda"
        ])
        f_out.flush()

    print(f"\n=== Running Suite: {suite_name} (dim={dim}x{dim}, {len(boards)} boards) ===", flush=True)
    for b_idx, board in enumerate(boards):
        init_md = manhattan_linear_conflict(board, dim)
        row_summary = []
        for m_name, m_list in methods:
            if (suite_name, b_idx, m_name) in completed:
                continue
            res = solve_sliding_tile_astar(
                init_board=board,
                dim=dim,
                mode=m_name,
                models=m_list,
                device=device,
                max_expansions=max_expansions,
                timeout=timeout,
            )
            writer.writerow([
                suite_name, dim, b_idx, f"{init_md:.1f}", m_name,
                int(res["solved"]), res["expansions"], res["plan_length"],
                f"{res['runtime']:.4f}", f"{res['mean_sigma']:.4f}", f"{res['mean_lambda']:.4f}"
            ])
            f_out.flush()
            tag = f"{res['expansions']}" if res["solved"] else "FAIL"
            row_summary.append(f"{m_name[:10]}:{tag}")

        if row_summary:
            print(f"[{suite_name} {b_idx+1:2d}/{len(boards)} MD={init_md:.0f}] " + " | ".join(row_summary), flush=True)

    f_out.close()


def run_calibration_and_generate_figures(
    paper_model: ChrestienHeuristicNet,
    ensemble_models: list,
    device: torch.device,
    csv_5x5: str,
    csv_ood: str,
):
    print("\n=== Generating Sliding-Tile Calibration & Benchmark Figures ===", flush=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Sample 400 states along exact walks from goal on 5x5 to evaluate Goal-Anchored Calibration & Uncertainty vs Error
    goal = get_goal_board(5)
    rng = np.random.default_rng(777)
    states, true_d = [], []
    for walk_idx in range(10):
        cur = goal.copy()
        pr, pc = -1, -1
        for d in range(1, 41):
            br, bc = np.argwhere(cur == 0)[0]
            cands = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = br + dr, bc + dc
                if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) != (pr, pc):
                    nb = cur.copy()
                    nb[br, bc], nb[nr, nc] = nb[nr, nc], 0
                    cands.append((manhattan_linear_conflict(nb, 5), nb, nr, nc))
            cands.sort(key=lambda x: x[0], reverse=True)
            idx = 0 if (rng.random() < 0.70 or len(cands) == 1) else int(rng.integers(1, len(cands)))
            pr, pc = br, bc
            cur = cands[idx][1]
            states.append(cur.copy())
            true_d.append(float(d))

    true_d = np.array(true_d, dtype=np.float64)
    s_t = encode_boards_25ch(states, device)
    g_t = encode_boards_25ch([goal], device)
    gb_t = g_t.expand(len(states), -1, -1, -1)

    M = len(ensemble_models)
    raw_ens = np.zeros((M, len(states)), dtype=np.float64)
    g_offs = np.zeros(M, dtype=np.float64)
    with torch.no_grad():
        for m_i, m in enumerate(ensemble_models):
            g_offs[m_i] = float(m(g_t, g_t)[0].item())
            raw_ens[m_i] = m(s_t, gb_t).cpu().numpy().astype(np.float64) - g_offs[m_i]

    # Zero-center & scale-normalize per model
    m_means = np.mean(raw_ens, axis=1, keepdims=True)
    zc_ens = raw_ens - (m_means - np.mean(raw_ens))
    # Map ensemble mean to physical step scale via linear fit on goal-anchored predictions
    mu_raw = np.mean(raw_ens, axis=0)
    slope = float(np.sum(mu_raw * true_d) / max(np.sum(mu_raw ** 2), 1e-6))
    mu_phys = mu_raw * slope
    sigma_phys = np.std(zc_ens * slope, axis=0)
    abs_err = np.abs(mu_phys - true_d)

    r_cal, p_cal = stats.pearsonr(true_d, mu_phys)
    r_err, p_err = stats.pearsonr(sigma_phys, abs_err)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=180)

    # Panel A: Goal-Anchored Calibration (True Cost-to-Go vs Predicted)
    ax = axes[0]
    sc = ax.scatter(true_d, mu_phys, c=sigma_phys, cmap="viridis", alpha=0.75, s=28, edgecolors="none")
    ax.plot([0, 42], [0, 42], "r--", lw=2, label="Ideal 1:1 Calibration")
    plt.colorbar(sc, ax=ax, label=r"Ensemble Uncertainty $\sigma_h(s)$")
    ax.set_title(f"A. Sliding-Tile Goal-Anchored Calibration\n(Pearson r = {r_cal:.4f}, MAE = {np.mean(abs_err):.2f} steps)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Walk Depth / Step Distance from Goal", fontsize=10)
    ax.set_ylabel(r"Goal-Anchored Ensemble Prediction $\hat{h}(s) - \hat{h}(g)$", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel B: Binned Epistemic Uncertainty vs Actual Heuristic Error
    ax = axes[1]
    bins = np.percentile(sigma_phys, np.linspace(0, 100, 6))
    bin_x, bin_y, bin_se = [], [], []
    for i in range(5):
        mask = (sigma_phys >= bins[i]) & (sigma_phys <= bins[i + 1] if i == 4 else sigma_phys < bins[i + 1])
        if np.sum(mask) > 2:
            bin_x.append(float(np.mean(sigma_phys[mask])))
            bin_y.append(float(np.mean(abs_err[mask])))
            bin_se.append(float(np.std(abs_err[mask]) / np.sqrt(np.sum(mask))))
    ax.errorbar(bin_x, bin_y, yerr=bin_se, fmt="-o", color="#d95f02", lw=2.5, markersize=8, capsize=5, label="Binned Mean Error ± 1 SE")
    ax.set_title(f"B. Uncertainty vs. Heuristic Error (24-Puzzle)\n(Pearson r = {r_err:.4f}, p = {p_err:.2e})", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Zero-Centered Ensemble Uncertainty $\sigma_h(s)$", fontsize=10)
    ax.set_ylabel(r"Absolute Prediction Error $|\hat{h}(s) - d(s)|$", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel C: Solve Rate & Mean Expansions Summary across Suites
    ax = axes[2]
    all_rows = []
    for p in (csv_5x5, csv_ood):
        if os.path.exists(p):
            with open(p, "r") as f:
                all_rows.extend(list(csv.DictReader(f)))

    if all_rows:
        method_order = [
            "classical_md_lc", "paper_baseline_lstar", "mc_dropout_conf_aware",
            "mc_dropout_risk_averse", "single_scratch_m0", "normal_ensemble",
            "conf_aware_ensemble", "risk_averse_ensemble"
        ]
        short_labels = [
            "MD+LC", "Paper L*", "MC-CA", "MC-RA",
            "Scratch M0", "Norm Ens", "Conf-Aware", "Risk-Averse"
        ]
        solve_rates = []
        for m in method_order:
            m_rows = [r for r in all_rows if r["method"] == m]
            sr = 100.0 * np.mean([int(r["solved"]) for r in m_rows]) if m_rows else 0.0
            solve_rates.append(sr)

        colors = ["#888888", "#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#a65628", "#e41a1c", "#1b9e77"]
        bars = ax.bar(short_labels, solve_rates, color=colors, edgecolor="black", alpha=0.88)
        for bar, sr in zip(bars, solve_rates):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.0, f"{sr:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_ylim(0, max(105.0, max(solve_rates) + 12.0))
        ax.set_ylabel("Overall Solve Rate (%) Across All Suites", fontsize=10)
        ax.set_title("C. Sliding-Tile Overall Solve Rate (5x5, 6x6, 7x7)", fontsize=11, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = "results/figures/fig9_sliding_tile_results_and_calibration.png"
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"Saved figure to {fig_path}")

    # Write comprehensive Markdown summary report
    report_path = "results/sliding_tile_benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Sliding-Tile Puzzle ($N$-Puzzle) Benchmark & Calibration Report\n\n")
        f.write(f"- **Goal-Anchored Calibration ($5\\times 5$):** Pearson $r = {r_cal:.4f}$ ($p = {p_cal:.2e}$), MAE $= {np.mean(abs_err):.2f}$ steps\n")
        f.write(f"- **Uncertainty vs. Heuristic Error:** Pearson $r = {r_err:.4f}$ ($p = {p_err:.2e}$)\n\n")

        suites = sorted(set(r["suite"] for r in all_rows))
        for s_name in suites:
            f.write(f"## Suite: `{s_name}`\n\n")
            f.write("| Method | Solved (%) | Mean Expansions (All) | Median Expansions (All) | Mean Expansions (Solved) | Mean Runtime (s) | Mean $\\sigma_h$ |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for m in method_order:
                s_rows = [r for r in all_rows if r["suite"] == s_name and r["method"] == m]
                if not s_rows:
                    continue
                solved_flags = [int(r["solved"]) for r in s_rows]
                exps_all = [int(r["expansions"]) for r in s_rows]
                exps_solved = [int(r["expansions"]) for r in s_rows if int(r["solved"]) == 1]
                rts = [float(r["runtime"]) for r in s_rows]
                sigs = [float(r["mean_sigma"]) for r in s_rows]
                sr_str = f"{100.0 * np.mean(solved_flags):.1f}% ({sum(solved_flags)}/{len(s_rows)})"
                m_sol = f"{np.mean(exps_solved):.1f}" if exps_solved else "N/A"
                f.write(
                    f"| `{m}` | **{sr_str}** | {np.mean(exps_all):.1f} | {np.median(exps_all):.1f} | "
                    f"{m_sol} | {np.mean(rts):.2f}s | {np.mean(sigs):.2f} |\n"
                )
            f.write("\n")
    print(f"Saved full Sliding-Tile report to {report_path}")


def main():
    t_total = time.time()
    device = torch.device(
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"=== Starting Overnight Sliding-Tile Research Pipeline on {device} ===", flush=True)

    # Phase B Step 1: Ensure all 5 Deep Ensemble models are trained
    train_sliding_tile_ensemble(
        num_models=5,
        epochs=12,
        batch_groups=16,
        lr=1.5e-4,
        out_dir="checkpoints/slidingtile",
    )

    # Load paper baseline model + 5 trained ensemble models
    paper_model = ChrestienHeuristicNet(dim=5, in_channels=50).to(device)
    paper_model.load_state_dict(torch.load("checkpoints/slidingtile/stile_paper_pytorch.pt", map_location=device))
    paper_model.eval()

    ensemble_models = []
    for m_idx in range(5):
        m = ChrestienHeuristicNet(dim=5, in_channels=50).to(device)
        m.load_state_dict(torch.load(f"checkpoints/slidingtile/stile_ens_m{m_idx}.pt", map_location=device))
        m.eval()
        ensemble_models.append(m)

    # Suite 1: 5x5 (24-Puzzle) In-Distribution Benchmark (50 boards from test/stile5.txt + held-out train/stile.txt)
    boards_test5 = load_sliding_tile_boards("Optimize-Planning-Heuristics-to-Rank/slidingtile/test/stile5.txt", 5)[:25]
    boards_train5_heldout = load_sliding_tile_boards("Optimize-Planning-Heuristics-to-Rank/slidingtile/train/stile.txt", 5)[-25:]
    boards_5x5 = boards_test5 + boards_train5_heldout

    csv_5x5 = "results/sliding_tile_5x5_benchmark.csv"
    run_benchmark_suite(
        suite_name="5x5_24puzzle_indist",
        boards=boards_5x5,
        dim=5,
        paper_model=paper_model,
        ensemble_models=ensemble_models,
        device=device,
        csv_path=csv_5x5,
        max_expansions=2000,
        timeout=14.0,
    )

    # Suite 2: 6x6 (35-Puzzle) OOD Scaling Benchmark (35 boards)
    boards_6x6 = generate_ood_benchmark_boards(dim=6, count=35, min_steps=22, max_steps=44)
    csv_ood = "results/sliding_tile_ood_6x6_7x7_benchmark.csv"
    run_benchmark_suite(
        suite_name="6x6_35puzzle_ood",
        boards=boards_6x6,
        dim=6,
        paper_model=paper_model,
        ensemble_models=ensemble_models,
        device=device,
        csv_path=csv_ood,
        max_expansions=1800,
        timeout=12.0,
    )

    # Suite 3: 7x7 (48-Puzzle) Extreme OOD Scaling Benchmark (35 boards)
    boards_7x7 = generate_ood_benchmark_boards(dim=7, count=35, min_steps=22, max_steps=44)
    run_benchmark_suite(
        suite_name="7x7_48puzzle_ood",
        boards=boards_7x7,
        dim=7,
        paper_model=paper_model,
        ensemble_models=ensemble_models,
        device=device,
        csv_path=csv_ood,
        max_expansions=1800,
        timeout=12.0,
    )

    # Phase C: Mechanistic Calibration & Publication Figure Generation
    run_calibration_and_generate_figures(
        paper_model=paper_model,
        ensemble_models=ensemble_models,
        device=device,
        csv_5x5=csv_5x5,
        csv_ood=csv_ood,
    )

    print(f"\n=== Overnight Sliding-Tile Pipeline Complete in {(time.time() - t_total)/60.0:.1f} minutes! ===", flush=True)


if __name__ == "__main__":
    main()
