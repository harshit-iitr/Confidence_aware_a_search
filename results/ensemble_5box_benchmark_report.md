# Benchmark Report: Three-Tier Search Ablation (5-Box Sokoban (Out-of-Distribution Benchmark))
### Decomposing Gains: Scratch Convergence vs. Ensembling vs. Confidence Gating

---

## 1. Experimental Overview
* **Domain Suite:** 5-Box Sokoban (Out-of-Distribution Benchmark)
* **Instances Evaluated:** 100 mazes
* **Hardware Platform:** xpu
* **Search Timeout:** 360.0s (6.0 minutes hard ceiling)
* **Expansion Limit:** 1000000 (Uncapped time-priority search)
* **Independent Physical Plan Replay:** 100% verified across all solved instances
* **Total Benchmark Runtime:** 232.19 minutes

---

## 2. Comprehensive Benchmark Summary Table

| Algorithm | Solve Rate | Plan Verified | Mean Expansions | Median Expansions | Mean Cost | Mean Time (ms) | Mean Confidence $\bar{\lambda}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single Scratch Model #0 (Ablation)** | 93.0% (93/100) | 100.0% | 8740.4 | 1436.0 | 37.5 | 28657.0 ms | 1.000 |
| **Normal 5-Model Ensemble A* (Pure Mean)** | 95.0% (95/100) | 100.0% | 1665.1 | 591.0 | 41.1 | 27176.3 ms | 1.000 |
| **Confidence-Aware 5-Model Ensemble A* (Ours)** | 95.0% (95/100) | 100.0% | **1614.9** | **549.0** | 41.0 | 26963.8 ms | 0.982 |

---

## 3. Methodological Ablation Breakdown
1. **Training & Convergence Factor (Paper NN $\to$ Scratch Model #0):**
   Examines how much search improvement is driven by your fresh 20,000-step training from scratch vs. the paper's legacy checkpoint.
2. **Variance Cancellation Factor (Scratch Model #0 $\to$ 5-Model Ensemble):**
   Measures the pure benefit of ensembling 5 bootstrap models (Breiman's variance reduction), eliminating erratic heuristic spikes without classical blending.
3. **Epistemic Gating Factor (Normal Ensemble $\to$ Confidence-Aware Ensemble):**
   Measures the specific reduction in dead-end exploration achieved by dynamically gating $\lambda(s)$ via multi-seed cross-model variance.
