import os
import csv
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def analyze_and_plot():
    csv_path = "results/sliding_tile_strict_paper_6min.csv"
    rows = list(csv.DictReader(open(csv_path, "r", newline="")))

    by_board = {}
    for r in rows:
        b = int(r["board_idx"])
        by_board.setdefault(b, {})[r["method"]] = r

    boards_all = sorted(by_board.keys())
    print(f"Total boards in CSV: {len(boards_all)}")

    methods = [
        "paper_baseline_lstar",
        "mc_dropout_risk_averse",
        "mc_dropout_conf_aware",
    ]
    labels = {
        "paper_baseline_lstar": "Paper Baseline (L*)",
        "mc_dropout_risk_averse": "MC-Dropout Risk-Averse (κ=0.50)",
        "mc_dropout_conf_aware": "MC-Dropout Conf-Aware (β=2.5)",
    }

    # Find which board paper_baseline_lstar failed on
    failed_base = [b for b in boards_all if int(by_board[b]["paper_baseline_lstar"]["solved"]) == 0]
    print(f"Boards where paper_baseline_lstar timed out (360s): {failed_base}")
    for b in failed_base:
        for m in methods:
            print(f"  Board {b} {m}: {by_board[b][m]}")

    common_solved = [
        b for b in boards_all
        if all(int(by_board[b][m]["solved"]) == 1 for m in methods)
    ]
    print(f"Common solved boards across all 3 methods: {len(common_solved)} / {len(boards_all)}")

    stats_summary = {}
    for m in methods:
        m_all = [by_board[b][m] for b in boards_all]
        m_sol = [r for r in m_all if int(r["solved"]) == 1]
        m_com = [by_board[b][m] for b in common_solved]

        p_all = np.array([int(r["paper_expansions"]) for r in m_all], dtype=np.float64)
        p_sol = np.array([int(r["paper_expansions"]) for r in m_sol], dtype=np.float64)
        p_com = np.array([int(r["paper_expansions"]) for r in m_com], dtype=np.float64)

        n_all = np.array([int(r["nodes_popped"]) for r in m_all], dtype=np.float64)
        n_sol = np.array([int(r["nodes_popped"]) for r in m_sol], dtype=np.float64)
        n_com = np.array([int(r["nodes_popped"]) for r in m_com], dtype=np.float64)

        l_sol = np.array([int(r["plan_length"]) for r in m_sol], dtype=np.float64)
        l_com = np.array([int(r["plan_length"]) for r in m_com], dtype=np.float64)
        rt_all = np.array([float(r["runtime"]) for r in m_all], dtype=np.float64)
        rt_sol = np.array([float(r["runtime"]) for r in m_sol], dtype=np.float64)
        sig_all = np.array([float(r["mean_sigma"]) for r in m_all], dtype=np.float64)
        lam_all = np.array([float(r["mean_lambda"]) for r in m_all], dtype=np.float64)

        stats_summary[m] = {
            "solved_count": len(m_sol),
            "solve_rate": 100.0 * len(m_sol) / len(m_all),
            "p_all": p_all, "p_sol": p_sol, "p_com": p_com,
            "n_all": n_all, "n_sol": n_sol, "n_com": n_com,
            "l_sol": l_sol, "l_com": l_com,
            "rt_all": rt_all, "rt_sol": rt_sol,
            "sig_all": sig_all, "lam_all": lam_all,
        }

    base_p_com = stats_summary["paper_baseline_lstar"]["p_com"]
    ra_p_com = stats_summary["mc_dropout_risk_averse"]["p_com"]
    ca_p_com = stats_summary["mc_dropout_conf_aware"]["p_com"]

    base_n_com = stats_summary["paper_baseline_lstar"]["n_com"]
    ra_n_com = stats_summary["mc_dropout_risk_averse"]["n_com"]
    ca_n_com = stats_summary["mc_dropout_conf_aware"]["n_com"]

    # Wilcoxon tests on common solved (N=199)
    w_ra, p_wilcox_ra = stats.wilcoxon(base_p_com, ra_p_com)
    w_ca, p_wilcox_ca = stats.wilcoxon(base_p_com, ca_p_com)
    w_ca_vs_ra, p_wilcox_ca_ra = stats.wilcoxon(ra_p_com, ca_p_com)

    t_ra, p_ttest_ra = stats.ttest_rel(base_p_com, ra_p_com)
    t_ca, p_ttest_ca = stats.ttest_rel(base_p_com, ca_p_com)

    ra_wins = int(np.sum(ra_p_com < base_p_com))
    ra_ties = int(np.sum(ra_p_com == base_p_com))
    ra_losses = int(np.sum(ra_p_com > base_p_com))

    ca_wins = int(np.sum(ca_p_com < base_p_com))
    ca_ties = int(np.sum(ca_p_com == base_p_com))
    ca_losses = int(np.sum(ca_p_com > base_p_com))

    print("\n=== SUMMARY ON ALL 200 BOARDS ===")
    for m in methods:
        s = stats_summary[m]
        print(
            f"{m:24s}: Solved={s['solved_count']}/200 ({s['solve_rate']:.1f}%) | "
            f"PaperExp(sol) Mean={np.mean(s['p_sol']):.1f}, Med={np.median(s['p_sol']):.1f}, 90th={np.percentile(s['p_sol'], 90):.1f}, 95th={np.percentile(s['p_sol'], 95):.1f} | "
            f"Popped(sol) Mean={np.mean(s['n_sol']):.1f}, Med={np.median(s['n_sol']):.1f} | "
            f"Len={np.mean(s['l_sol']):.1f} | Time(all)={np.mean(s['rt_all']):.2f}s"
        )

    print("\n=== HEAD-TO-HEAD ON 199 COMMON SOLVED BOARDS ===")
    print(
        f"Risk-Averse vs Baseline: Mean {np.mean(ra_p_com):.1f} vs {np.mean(base_p_com):.1f} "
        f"({100*(np.mean(ra_p_com)-np.mean(base_p_com))/np.mean(base_p_com):+.2f}%), "
        f"Median {np.median(ra_p_com):.1f} vs {np.median(base_p_com):.1f} "
        f"({100*(np.median(ra_p_com)-np.median(base_p_com))/np.median(base_p_com):+.2f}%), "
        f"W/T/L={ra_wins}/{ra_ties}/{ra_losses} ({100*ra_wins/(ra_wins+ra_losses):.1f}% win), "
        f"Wilcoxon p={p_wilcox_ra:.4e}, paired-t p={p_ttest_ra:.4e}"
    )
    print(
        f"Conf-Aware vs Baseline : Mean {np.mean(ca_p_com):.1f} vs {np.mean(base_p_com):.1f} "
        f"({100*(np.mean(ca_p_com)-np.mean(base_p_com))/np.mean(base_p_com):+.2f}%), "
        f"Median {np.median(ca_p_com):.1f} vs {np.median(base_p_com):.1f} "
        f"({100*(np.median(ca_p_com)-np.median(base_p_com))/np.median(base_p_com):+.2f}%), "
        f"W/T/L={ca_wins}/{ca_ties}/{ca_losses} ({100*ca_wins/(ca_wins+ca_losses):.1f}% win), "
        f"Wilcoxon p={p_wilcox_ca:.4e}, paired-t p={p_ttest_ca:.4e}"
    )

    # Generate 4-panel publication figure
    os.makedirs("results/figures", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=180)

    # Panel A: Cumulative Solved Cactus Plot (Expansions Budget vs Solved Boards out of 200)
    ax = axes[0, 0]
    colors = {
        "paper_baseline_lstar": "#d95f02",
        "mc_dropout_risk_averse": "#7570b3",
        "mc_dropout_conf_aware": "#1b9e77",
    }
    for m in methods:
        s_exps = np.sort(stats_summary[m]["p_sol"])
        y_vals = np.arange(1, len(s_exps) + 1)
        ax.step(s_exps, y_vals, where="post", lw=2.5, color=colors[m], label=f"{labels[m]} ({len(s_exps)}/200)")
    ax.set_xscale("log")
    ax.set_xlabel("Paper State Expansions (Log Scale)", fontsize=11)
    ax.set_ylabel("Cumulative Boards Solved (out of 200)", fontsize=11)
    ax.set_title("A. Cumulative Solved Profile on 24-Puzzle (200 Test Boards)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9.5)

    # Panel B: Head-to-Head Scatter (Paper Baseline vs MC-Dropout Confidence-Aware)
    ax = axes[0, 1]
    max_v = max(np.max(base_p_com), np.max(ca_p_com)) * 1.15
    min_v = min(np.min(base_p_com), np.min(ca_p_com)) * 0.85
    ax.scatter(base_p_com, ca_p_com, c="#1b9e77", alpha=0.65, s=36, edgecolors="black", linewidth=0.4, label=f"Conf-Aware: {ca_wins}W / {ca_losses}L (p={p_wilcox_ca:.2e})")
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=2, label="1:1 Parity Line (Below = Ours Faster)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel("Paper Baseline L* Expansions (Log)", fontsize=11)
    ax.set_ylabel("MC-Dropout Conf-Aware Expansions (Log)", fontsize=11)
    ax.set_title("B. Head-to-Head: Conf-Aware vs. Paper Baseline (N=199)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9.5)

    # Panel C: Head-to-Head Scatter (Paper Baseline vs MC-Dropout Risk-Averse)
    ax = axes[1, 0]
    max_v2 = max(np.max(base_p_com), np.max(ra_p_com)) * 1.15
    min_v2 = min(np.min(base_p_com), np.min(ra_p_com)) * 0.85
    ax.scatter(base_p_com, ra_p_com, c="#7570b3", alpha=0.65, s=36, edgecolors="black", linewidth=0.4, label=f"Risk-Averse: {ra_wins}W / {ra_losses}L (p={p_wilcox_ra:.2e})")
    ax.plot([min_v2, max_v2], [min_v2, max_v2], "r--", lw=2, label="1:1 Parity Line (Below = Ours Faster)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min_v2, max_v2)
    ax.set_ylim(min_v2, max_v2)
    ax.set_xlabel("Paper Baseline L* Expansions (Log)", fontsize=11)
    ax.set_ylabel("MC-Dropout Risk-Averse Expansions (Log)", fontsize=11)
    ax.set_title("C. Head-to-Head: Risk-Averse vs. Paper Baseline (N=199)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9.5)

    # Panel D: Bar Breakdown of Median, Mean, and 90th Percentile Tail Expansions (Common Solved N=199)
    ax = axes[1, 1]
    metrics_names = ["Median\nExpansions", "Mean\nExpansions", "90th Percentile\n(Hard Tail)", "95th Percentile\n(Extreme Tail)"]
    x = np.arange(len(metrics_names))
    width = 0.26

    for idx_m, m in enumerate(methods):
        arr = stats_summary[m]["p_com"]
        vals = [np.median(arr), np.mean(arr), np.percentile(arr, 90), np.percentile(arr, 95)]
        rects = ax.bar(x + (idx_m - 1) * width, vals, width, label=labels[m], color=colors[m], edgecolor="black", alpha=0.88)
        for rect, v in zip(rects, vals):
            ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height() + 350, f"{v:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
    ax.set_ylabel("Paper State Expansions (Lower is Better)", fontsize=11)
    ax.set_ylim(0, max(np.percentile(base_p_com, 95), np.percentile(ra_p_com, 95)) * 1.18)
    ax.set_title("D. Expansion Distribution Breakdown (N=199 Common Boards)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig_out = "results/figures/fig9_sliding_tile_strict_200boards.png"
    plt.savefig(fig_out, dpi=180)
    plt.close(fig)
    print(f"\nSaved 4-panel figure to {fig_out}")

    artifact_fig = r"C:\Users\ASUS\.gemini\antigravity\brain\4d0676d1-5473-4f4f-a0d9-e2940572a93d\fig9_sliding_tile_strict_200boards.png"
    shutil.copyfile(fig_out, artifact_fig)

    # Write comprehensive Markdown report
    report_path = "results/sliding_tile_strict_paper_6min_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Sliding-Tile ($5\\times 5$ 24-Puzzle) Strict Paper A* Benchmark Report (200 Boards)\n\n")
        f.write("## 1. Experimental Protocol\n")
        f.write("- **Domain:** $5\\times 5$ Sliding-Tile Puzzle (**24-Puzzle**, state space $\\sim 1.55 \\times 10^{25}$), all 200 test boards from `Optimize-Planning-Heuristics-to-Rank/slidingtile/test/stile5.txt`.\n")
        f.write("- **Model:** Exact converted pre-trained `stile` checkpoint (`checkpoints/slidingtile/stile_paper_pytorch.pt`) provided by Chrestien et al. (zero weight retraining).\n")
        f.write("- **Search Implementation:** Strict `test_on_Astar.py` implementation (`openSet` + `closedSet` with closed-state reopening, **no expansion limit**, **360.0s / 6-minute timeout** per board).\n")
        f.write("- **Hardware:** Intel Arc GPU (`xpu`).\n\n")

        f.write("## 2. Overall Performance Across All 200 Test Boards\n\n")
        f.write("| Method | Solved (%) | Mean Paper Exp (Solved) | Median Paper Exp (Solved) | 90th-% Exp (Solved) | 95th-% Exp (Solved) | Mean Nodes Popped | Mean Plan Length | Mean Runtime (All) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for m in methods:
            s = stats_summary[m]
            f.write(
                f"| **{labels[m]}** (`{m}`) | **{s['solve_rate']:.1f}% ({s['solved_count']}/200)** | "
                f"**{np.mean(s['p_sol']):.1f}** | **{np.median(s['p_sol']):.1f}** | "
                f"{np.percentile(s['p_sol'], 90):.1f} | {np.percentile(s['p_sol'], 95):.1f} | "
                f"{np.mean(s['n_sol']):.1f} | {np.mean(s['l_sol']):.1f} | {np.mean(s['rt_all']):.2f}s |\n"
            )

        f.write("\n## 3. Controlled Head-to-Head Analysis on 199 Mutually Solved Boards\n\n")
        f.write("| Comparison Pair | Mean Expansions | Median Expansions | 90th-% Tail | 95th-% Tail | Win / Tie / Loss | Win Rate | Wilcoxon Signed-Rank $p$ | Paired $t$-Test $p$ |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(
            f"| **MC-Dropout Conf-Aware vs. Paper Baseline** | "
            f"**{np.mean(ca_p_com):.1f}** vs. {np.mean(base_p_com):.1f} (**{100*(np.mean(ca_p_com)-np.mean(base_p_com))/np.mean(base_p_com):+.2f}%**) | "
            f"**{np.median(ca_p_com):.1f}** vs. {np.median(base_p_com):.1f} (**{100*(np.median(ca_p_com)-np.median(base_p_com))/np.mean(base_p_com):+.2f}%**) | "
            f"**{np.percentile(ca_p_com, 90):.1f}** vs. {np.percentile(base_p_com, 90):.1f} (**{100*(np.percentile(ca_p_com, 90)-np.percentile(base_p_com, 90))/np.percentile(base_p_com, 90):+.2f}%**) | "
            f"**{np.percentile(ca_p_com, 95):.1f}** vs. {np.percentile(base_p_com, 95):.1f} (**{100*(np.percentile(ca_p_com, 95)-np.percentile(base_p_com, 95))/np.percentile(base_p_com, 95):+.2f}%**) | "
            f"**{ca_wins}W / {ca_ties}T / {ca_losses}L** | **{100*ca_wins/(ca_wins+ca_losses):.1f}%** | **`{p_wilcox_ca:.4e}`** | `{p_ttest_ca:.4e}` |\n"
        )
        f.write(
            f"| **MC-Dropout Risk-Averse vs. Paper Baseline** | "
            f"**{np.mean(ra_p_com):.1f}** vs. {np.mean(base_p_com):.1f} (**{100*(np.mean(ra_p_com)-np.mean(base_p_com))/np.mean(base_p_com):+.2f}%**) | "
            f"**{np.median(ra_p_com):.1f}** vs. {np.median(base_p_com):.1f} (**{100*(np.median(ra_p_com)-np.median(base_p_com))/np.median(base_p_com):+.2f}%**) | "
            f"**{np.percentile(ra_p_com, 90):.1f}** vs. {np.percentile(base_p_com, 90):.1f} (**{100*(np.percentile(ra_p_com, 90)-np.percentile(base_p_com, 90))/np.percentile(base_p_com, 90):+.2f}%**) | "
            f"**{np.percentile(ra_p_com, 95):.1f}** vs. {np.percentile(base_p_com, 95):.1f} (**{100*(np.percentile(ra_p_com, 95)-np.percentile(base_p_com, 95))/np.percentile(base_p_com, 95):+.2f}%**) | "
            f"**{ra_wins}W / {ra_ties}T / {ra_losses}L** | **{100*ra_wins/(ra_wins+ra_losses):.1f}%** | **`{p_wilcox_ra:.4e}`** | `{p_ttest_ra:.4e}` |\n"
        )

        if failed_base:
            b_fail = failed_base[0]
            r_b = by_board[b_fail]["paper_baseline_lstar"]
            r_ra = by_board[b_fail]["mc_dropout_risk_averse"]
            r_ca = by_board[b_fail]["mc_dropout_conf_aware"]
            f.write(f"\n## 4. Deep-Dive on Board #{b_fail} (Where Paper Baseline Timed Out at 360s)\n\n")
            f.write(f"- **Board #{b_fail}** (`init_md_lc = {r_b['init_md_lc']}`):\n")
            f.write(f"  - `paper_baseline_lstar`: **TIMEOUT (`360.01s`)** after expanding **{r_b['paper_expansions']}** states (`{r_b['nodes_popped']}` popped) — trapped in a deep deceptive plateau.\n")
            f.write(f"  - `mc_dropout_risk_averse`: **SOLVED** in **{r_ra['paper_expansions']}** paper expansions (`{r_ra['nodes_popped']}` popped, plan length `{r_ra['plan_length']}`, `{float(r_ra['runtime']):.2f}s`).\n")
            f.write(f"  - `mc_dropout_conf_aware`: **SOLVED** in **{r_ca['paper_expansions']}** paper expansions (`{r_ca['nodes_popped']}` popped, plan length `{r_ca['plan_length']}`, `{float(r_ca['runtime']):.2f}s`).\n")

    print(f"Updated full report at {report_path}")


if __name__ == "__main__":
    analyze_and_plot()
