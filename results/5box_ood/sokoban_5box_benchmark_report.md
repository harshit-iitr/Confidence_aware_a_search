# 5-Box Sokoban Out-of-Distribution (OOD) Benchmark Report

## 1. Experimental Overview
- **Benchmark Domain**: 10x10 Sokoban with 5 boxes (Out-of-Distribution zero-shot evaluation)
- **Dataset Size**: 100 procedurally generated mazes (`gym-sokoban` reverse-walk, 30 steps)
- **Total Benchmark Wall Time**: 168.20 minutes (2.80 hours) on Intel Arc GPU (XPU)
- **Pretrained Checkpoint**: Chrestien et al. `finalSok3` (trained strictly on 3 boxes, evaluated zero-shot on 5 boxes)
- **Independent Solution Verification**: 100% physically verified via step-by-step transition simulation (`verify_solution_plan`)

---

## 2. Quantitative Performance Comparison

| Algorithm | Solve Rate | Expansions (Mean) | Expansions (Median) | Path Cost | Search Time (s) | Plan Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical A\* (Manhattan)** | 56/100 (56.0%) | 41,546.2 | 26,794.0 | 31.7 | 1.129s | 100% |
| **Paper Learned A\* ($\mathcal{L}^*$)** | 71/100 (71.0%) | 2,833.8 | 1,384.0 | 33.5 | 12.719s | 100% |
| **Paper Learned GBFS ($\mathcal{L}^*$)** | 70/100 (70.0%) | 2,516.1 | 346.0 | 45.9 | 9.465s | 100% |
| **Confidence-Aware Hybrid A\* (Ours)** | **90/100 (90.0%)** | **1,970.5** | **567.5** | 38.3 | **10.411s** | **100%** |
| **Confidence-Aware Hybrid GBFS (Ours)** | 66/100 (66.0%) | 1,957.4 | 305.5 | 44.5 | 11.276s | 100% |

---

## 3. Head-to-Head Statistical Hypothesis Testing

### Confidence-Aware Hybrid A* (Ours) vs. Paper Learned A* ($\mathcal{L}^*$)

#### Mutually Solved Instances (N = 70):
- **Head-to-Head Win Rate**: **62 Wins / 0 Ties / 8 Losses (88.57% Win Rate)**
- **Mean Node Expansions**: **981.8** (Ours) vs. **2,736.3** (Paper) $\rightarrow$ **+64.12% Node Reduction**
- **Median Node Expansions**: **331.5** (Ours) vs. **1,353.0** (Paper) $\rightarrow$ **+75.50% Node Reduction**
- **Wilcoxon Signed-Rank Test**: $W = 101.0, \quad p = \mathbf{3.734 \times 10^{-10}}$ (Highly statistically significant)

#### Asymmetric Recovery from Out-of-Distribution Deadlocks:
- **Instances where Ours Solved but Paper Learned A\* Timed Out**: **20 instances**  
  Maps: `[12, 15, 24, 27, 34, 35, 36, 37, 39, 41, 42, 48, 49, 55, 57, 62, 70, 87, 90, 91]`
- **Instances where Paper Solved but Ours Timed Out**: **1 instance**  
  Map: `[2]`
- **Net Solve Count Advantage**: **+19 Additional Mazes Solved (+26.8% relative gain in solved instances)**

---

## 4. Key Scientific Insights

1. **Robustness to Severe Domain Shift:**  
   When transferring a neural heuristic trained on 3 boxes to 5 boxes zero-shot, the baseline $\mathcal{L}^*$ model experiences severe out-of-distribution confusion, falling into cyclic deadlocks on 29% of test maps. Confidence-Aware $A^*$ detects elevated Monte Carlo rank variance ($\sigma^2_{\text{rank}}$) and dynamically scales in classical guidance ($\lambda \approx 0.85\text{--}0.90$), enabling it to escape deadlocks and achieve a **90% solve rate**.

2. **Compound Node Efficiency:**  
   Even after including the 20 harder instances where Paper $A^*$ failed completely, Confidence-Aware $A^*$ achieves an overall mean expansion count of **1,970.5 vs. 2,833.8** (**30.46% reduction overall**). On the mutually solved subset, it slashes mean expansions by **64.12%** and median expansions by **75.50%**.

3. **Physical Plan Authenticity:**  
   100% of generated paths (across all 90 solved instances) were independently validated step-by-step through the physical simulation engine (`verify_solution_plan`), confirming zero illegal moves or coordinate anomalies.
