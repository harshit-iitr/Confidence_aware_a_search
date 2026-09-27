# Benchmark Report: 5-Box OOD Calibrated Experiments
### MC-Dropout on Scratch Model #0 vs. Calibrated Epistemic Ensemble Gating

---

## 1. Experimental Overview
* **Dataset:** 5-Box Sokoban (100 Out-of-Distribution Mazes)
* **Compute Platform:** xpu
* **Search Timeout:** 360.0s (6.0 minutes hard ceiling)
* **Expansion Limit:** 1000000 (Uncapped time-priority search)
* **Calibrated Gating Hyperparameters:** Standard Deviation Mode, $\\beta = 2.5$, $\\lambda_{\\min} = 0.5$
* **Total Runtime:** 263.85 minutes

---

## 2. Benchmark Summary Table

| Algorithm | Solve Rate | Plan Verified | Mean Expansions | Median Expansions | 90th % Expansions | Mean Cost | Mean Time (ms) | Mean $\\bar{\\lambda}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MC-Dropout (Scratch Model #0)** | 77.0% (77/100) | 100.0% | 7906.9 | 3042.0 | 20360.2 | 36.4 | 39908.0 ms | 0.803 |
| **Calibrated Conf-Aware Ensemble (Ours)** | 94.0% (94/100) | 100.0% | **1553.5** | **513.0** | **4697.7** | 40.8 | 24604.6 ms | 0.930 |

---

## 3. Comparative Context vs. Reference Baselines
* **Paper Learned A\* (Deterministic)**: 71.0% Solved | Mean Exp: 2,833.8 | Median Exp: 1,384.0
* **MC-Dropout on Paper Model**: 90.0% Solved | Mean Exp: 1,970.5 | Median Exp: 567.5
* **Scratch Model #0 (Deterministic)**: 93.0% Solved | Mean Exp: 8,740.4 | Median Exp: 1,436.0
* **Normal 5-Model Ensemble (Pure Mean)**: 95.0% Solved | Mean Exp: 1,665.1 | Median Exp: 591.0
