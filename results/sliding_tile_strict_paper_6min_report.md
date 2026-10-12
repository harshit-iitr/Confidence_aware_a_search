# Sliding-Tile ($5\times 5$ 24-Puzzle) Strict Paper A* Benchmark Report (200 Boards)

## 1. Experimental Protocol
- **Domain:** $5\times 5$ Sliding-Tile Puzzle (**24-Puzzle**, state space $\sim 1.55 \times 10^{25}$), all 200 test boards from `Optimize-Planning-Heuristics-to-Rank/slidingtile/test/stile5.txt`.
- **Model:** Exact converted pre-trained `stile` checkpoint (`checkpoints/slidingtile/stile_paper_pytorch.pt`) provided by Chrestien et al. (zero weight retraining).
- **Search Implementation:** Strict `test_on_Astar.py` implementation (`openSet` + `closedSet` with closed-state reopening, **no expansion limit**, **360.0s / 6-minute timeout** per board).
- **Hardware:** Intel Arc GPU (`xpu`).

## 2. Overall Performance Across All 200 Test Boards

| Method | Solved (%) | Mean Paper Exp (Solved) | Median Paper Exp (Solved) | 90th-% Exp (Solved) | 95th-% Exp (Solved) | Mean Nodes Popped | Mean Plan Length | Mean Runtime (All) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Paper Baseline (L*)** (`paper_baseline_lstar`) | **99.5% (199/200)** | **8548.1** | **5841.0** | 17119.4 | 23909.8 | 3762.2 | 136.7 | 14.01s |
| **MC-Dropout Risk-Averse (κ=0.50)** (`mc_dropout_risk_averse`) | **100.0% (200/200)** | **6857.1** | **4756.5** | 14506.1 | 20968.3 | 3005.2 | 139.9 | 24.29s |
| **MC-Dropout Conf-Aware (β=2.5)** (`mc_dropout_conf_aware`) | **100.0% (200/200)** | **5329.0** | **3357.5** | 11579.9 | 14712.1 | 2365.0 | 137.2 | 19.38s |

## 3. Controlled Head-to-Head Analysis on 199 Mutually Solved Boards

| Comparison Pair | Mean Expansions | Median Expansions | 90th-% Tail | 95th-% Tail | Win / Tie / Loss | Win Rate | Wilcoxon Signed-Rank $p$ | Paired $t$-Test $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MC-Dropout Conf-Aware vs. Paper Baseline** | **5327.7** vs. 8548.1 (**-37.67%**) | **3350.0** vs. 5841.0 (**-29.14%**) | **11607.8** vs. 17119.4 (**-32.20%**) | **14822.3** vs. 23909.8 (**-38.01%**) | **141W / 0T / 58L** | **70.9%** | **`5.3579e-10`** | `2.2241e-06` |
| **MC-Dropout Risk-Averse vs. Paper Baseline** | **6752.0** vs. 8548.1 (**-21.01%**) | **4732.0** vs. 5841.0 (**-18.99%**) | **13227.0** vs. 17119.4 (**-22.74%**) | **18782.3** vs. 23909.8 (**-21.45%**) | **115W / 0T / 84L** | **57.8%** | **`2.8829e-03`** | `9.7139e-03` |

## 4. Deep-Dive on Board #25 (Where Paper Baseline Timed Out at 360s)

- **Board #25** (`init_md_lc = 90.0`):
  - `paper_baseline_lstar`: **TIMEOUT (`360.01s`)** after expanding **221858** states (`96461` popped) — trapped in a deep deceptive plateau.
  - `mc_dropout_risk_averse`: **SOLVED** in **27768** paper expansions (`12453` popped, plan length `164`, `100.95s`).
  - `mc_dropout_conf_aware`: **SOLVED** in **5595** paper expansions (`2421` popped, plan length `158`, `20.23s`).
