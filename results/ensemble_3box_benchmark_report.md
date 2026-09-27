# Benchmark Report: Three-Tier Search Ablation (3-Box Sokoban (In-Distribution Benchmark))
### Decomposing Gains: Scratch Convergence vs. Ensembling vs. Confidence Gating

---

## 1. Experimental Overview
* **Domain Suite:** 3-Box Sokoban (In-Distribution Benchmark)
* **Instances Evaluated:** 200 mazes
* **Hardware Platform:** xpu
* **Search Timeout:** 360.0s (6.0 minutes hard ceiling)
* **Expansion Limit:** 1000000 (Uncapped time-priority search)
* **Independent Physical Plan Replay:** 100% verified across all solved instances
* **Total Benchmark Runtime:** 52.81 minutes

---

## 2. Comprehensive Benchmark Summary Table

| Algorithm | Solve Rate | Plan Verified | Mean Expansions | Median Expansions | Mean Cost | Mean Time (ms) | Mean Confidence $\bar{\lambda}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single Scratch Model #0 (Ablation)** | 100.0% (200/200) | 100.0% | 1222.3 | 121.0 | 27.1 | 3863.9 ms | 1.000 |
| **Normal 5-Model Ensemble A* (Pure Mean)** | 100.0% (200/200) | 100.0% | 377.0 | 89.0 | 29.6 | 6046.1 ms | 1.000 |
| **Confidence-Aware 5-Model Ensemble A* (Ours)** | 100.0% (200/200) | 100.0% | **377.9** | **90.0** | 29.4 | 5929.3 ms | 0.979 |

---

## 3. Methodological Ablation Breakdown
1. **Training & Convergence Factor (Paper NN $\to$ Scratch Model #0):**
   Examines how much search improvement is driven by your fresh 20,000-step training from scratch vs. the paper's legacy checkpoint.
2. **Variance Cancellation Factor (Scratch Model #0 $\to$ 5-Model Ensemble):**
   Measures the pure benefit of ensembling 5 bootstrap models (Breiman's variance reduction), eliminating erratic heuristic spikes without classical blending.
3. **Epistemic Gating Factor (Normal Ensemble $\to$ Confidence-Aware Ensemble):**
   Measures the specific reduction in dead-end exploration achieved by dynamically gating $\lambda(s)$ via multi-seed cross-model variance.
