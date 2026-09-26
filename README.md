# Confidence-Aware Rank-Blended Heuristic Search

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(XPU%2FCUDA)-EE4C2C.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Intel%20Arc%20GPU%20(XPU)%20%2F%20CUDA-0071C5.svg)](https://www.intel.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An adaptive, confidence-aware heuristic search framework that dynamically estimates the reliability of learned neural ranking heuristics online and modulates the blend between learned and classical admissible heuristics.

This project builds upon, replicates, and rigorously evaluates against:
> **Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal**  
> *Chrestien et al., Advances in Neural Information Processing Systems (NeurIPS 2023).*

---

## 1. Motivation & Core Concept

Standard learned heuristics trained via pairwise ranking losses ($\mathcal{L}^*$) excel at ordering states along promising trajectories, but exhibit two critical vulnerabilities:
1. **Uncalibrated Scales in A\*:** Ranking models output arbitrary scalars without physical cost calibration, causing severe scale distortions when directly computing $f(s) = g(s) + h_\theta(s)$.
2. **Catastrophic Deadlocks on Ambiguous / OOD States:** When traversing out-of-distribution (OOD) or unfamiliar maze topologies, deterministic neural heuristics make overconfident erroneous decisions, trapping search in extensive deadlocks.

**Our Proposed Framework: Confidence-Aware Rank-Blended Search**
* **Test-Time Monte Carlo Dropout:** Samples $M=5$ stochastic predictions in a single parallel tensor pass to quantify epistemic uncertainty online without training multiple models from scratch.
* **Global Percentile Tracking ($\mathcal{H}_m, \mathcal{H}_{class}$):** Maps both the uncalibrated neural heuristics and the classical heuristic into a normalized rank-percentile space $[0, 1]$ in $O(\log V)$ per expansion.
* **Confidence Gating Function ($\lambda(s)$):** Computes rank variance $\sigma^2_{rank}(s)$ and modulates confidence dynamically:
  $$\lambda(s) = \lambda_{min} + (1 - \lambda_{min}) \cdot \exp\big(-C_{var} \cdot \sigma^2_{rank}(s)\big)$$
* **Rank-Space Convex Blend:** Converts the convex combination $p_{blend}(s)$ back to cost space via a static scale constant $C$:
  $$p_{blend}(s) = \lambda(s)\bar{p}(s) + \big(1 - \lambda(s)\big) p_{class}(s), \qquad h_{blend}(s) = p_{blend}(s) \cdot C$$

---

## 2. Architecture & Pipeline

```
                            Current State s
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
      Classical Admissible Heuristic      Pre-trained Network f_θ(s)
              h_class(s)                  with Test-Time MC Dropout
                 │                                 │
          [Rank in H_class]               [M Stochastic Passes]
                 │                                 │
        p_class(s) ∈ [0, 1]              {h_1(s), ..., h_M(s)}
                 │                                 │
                 │                          [Rank in {H_m}]
                 │                                 │
                 │                       {p_1(s), ..., p_M(s)}
                 │                                 │
                 │                    ┌────────────┴────────────┐
                 │                    ▼                         ▼
                 │             Mean Rank p̄(s)          Rank Variance σ²(s)
                 │                    │                         │
                 │                    │                         ▼
                 │                    │                  Confidence Gate
                 │                    │                 λ(s) ∈ [λ_min, 1]
                 │                    │                         │
                 └────────────────────┼─────────────────────────┘
                                      ▼
                       Rank-Space Convex Blend:
                p_blend(s) = λ(s)·p̄(s) + (1 - λ(s))·p_class(s)
                                      │
                                      ▼ [Multiply by Static Constant C]
                                 h_blend(s)
                                      │
                                      ▼
                            f(s) = g(s) + h_blend(s)
                                      │
                                      ▼
                            [Push to Open Min-Heap]
```

---

## 3. Empirical Results: 3-Box In-Distribution Rigorous Benchmark

Comprehensive evaluation across **200 clean test instances** (matching Chrestien et al. NeurIPS 2023 `states10test.txt`) with a rigorous **600.0-second timeout budget per instance**, running candidate neighbor batching on an Intel Arc GPU (XPU). Every single solution plan was **100% verified** by an independent transition simulator.

| Algorithm | Solve Rate | Node Expansions (Mean) | Node Expansions (Median) | Mean Path Cost | Mean Time (s) | Plan Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical A\* (Manhattan)** | 97.5% | 11,133.3 | 487.0 | 20.2 | 0.052s | 100% |
| **Paper Learned A\* ($\mathcal{L}^*$)** | 100.0% | 1,262.9 | 224.5 | 23.3 | 4.398s | 100% |
| **Paper Learned GBFS ($\mathcal{L}^*$)** | 100.0% | 944.4 | 43.0 | 26.6 | 1.839s | 100% |
| **Fixed 50/50 Hybrid A\*** | 100.0% | 762.2 | 155.5 | 25.1 | 4.542s | 100% |
| **Fixed 50/50 Hybrid GBFS** | 100.0% | 1,561.0 | 93.0 | 25.7 | 4.793s | 100% |
| **Confidence-Aware Hybrid A\* (Ours)** | **100.0%** | **596.0** | **92.0** | 25.4 | 4.090s | **100%** |
| **Confidence-Aware Hybrid GBFS (Ours)** | **100.0%** | 1,241.2 | 68.5 | 26.0 | 4.015s | **100%** |

### Statistical Significance & Head-to-Head Comparison:
* **Node Expansion Reduction:** **52.8% fewer expansions** compared to Paper Learned A\* ($p = 4.74 \times 10^{-22}$ via Wilcoxon signed-rank test).
* **Median Node Reduction:** **59.0% reduction** in median expansions (92.0 vs 224.5).
* **Head-to-Head Win Rate:** **82.0%** (164 Wins, 4 Ties, 32 Losses) against Paper Learned A\*.
* **Mean Gating Confidence:** $\bar{\lambda} = 0.902$, successfully detecting and avoiding neural deadlocks.

---

## 4. 5-Box Out-of-Distribution (OOD) Generalization Benchmark

To test robustness under domain shift and extreme branching complexity, the network trained exclusively on 3-box mazes is evaluated **zero-shot** on procedurally generated **5-box Sokoban mazes** (100 instances, evaluated on Intel Arc GPU).

| Algorithm | Solve Rate | Node Expansions (Mean) | Node Expansions (Median) | Mean Path Cost | Mean Time (s) | Plan Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical A\* (Manhattan)** | 56.0% | 41,546.2 | 26,794.0 | 31.7 | 1.129s | 100% |
| **Paper Learned A\* ($\mathcal{L}^*$)** | 71.0% | 2,833.8 | 1,384.0 | 33.5 | 12.719s | 100% |
| **Paper Learned GBFS ($\mathcal{L}^*$)** | 70.0% | 2,516.1 | 346.0 | 45.9 | 9.465s | 100% |
| **Confidence-Aware Hybrid A\* (Ours)** | **90.0%** | **1,970.5** | **567.5** | 38.3 | **10.411s** | **100%** |
| **Confidence-Aware Hybrid GBFS (Ours)** | 66.0% | 1,957.4 | 305.5 | 44.5 | 11.276s | 100% |

### Key OOD Findings:
* **Solve Rate Advantage Under Bounded Budget:** Paper Learned A\* suffers from out-of-distribution deadlocks, dropping to 71.0%. **Confidence-Aware Hybrid A\* maintains a 90.0% solve rate (+19.0% absolute improvement, +26.8% relative gain)**, successfully recovering solutions on 20 out of 29 deadlocked maps.
* **Node Reduction on Mutually Solved Mazes ($N=70$):** **64.12% fewer mean expansions** (981.8 vs 2,736.3) and **75.50% fewer median expansions** (331.5 vs 1,353.0) with an **88.6% head-to-head win rate** ($p = 3.73 \times 10^{-10}$ Wilcoxon).

---

## 5. Training Deep Ensembles (Bagging $M=5$)

To evaluate multi-model epistemic uncertainty as an alternative to test-time MC Dropout:

```bash
# Train Model Member #0 with 80% bootstrap subsampling (on GPU / T4 / RTX 4050)
python tools/train_deep_ensemble.py --model_id 0 --bagging_ratio 0.80 --episodes 2500 --output_dir checkpoints/ensemble

# Train remaining ensemble members (Models 1 to 4)
python tools/train_deep_ensemble.py --model_id 1 --bagging_ratio 0.80 --episodes 2500 --output_dir checkpoints/ensemble
python tools/train_deep_ensemble.py --model_id 2 --bagging_ratio 0.80 --episodes 2500 --output_dir checkpoints/ensemble
python tools/train_deep_ensemble.py --model_id 3 --bagging_ratio 0.80 --episodes 2500 --output_dir checkpoints/ensemble
python tools/train_deep_ensemble.py --model_id 4 --bagging_ratio 0.80 --episodes 2500 --output_dir checkpoints/ensemble

# Or train all 5 models sequentially
python tools/train_deep_ensemble.py --num_models 5 --bagging_ratio 0.80 --episodes 2500

# Benchmark the trained ensemble
python benchmarks/run_ensemble_benchmark.py --checkpoint_dir checkpoints/ensemble --dataset data/sokoban_5box_test.txt
```

---

## 6. Repository Structure

```
Confidence_aware_a_search/
├── README.md                          <- Project documentation and published benchmark results
├── requirements.txt                   <- Environment dependencies
├── finalSok3_pytorch.pt               <- Pretrained PyTorch weights (converted from NeurIPS 2023)
│
├── src/                               <- Core library modules
│   ├── __init__.py
│   ├── sokoban_env.py                 <- Sokoban grid simulation, legal transitions, tensor encoding
│   ├── classical_heuristics.py        <- Classical admissible heuristics (Manhattan distance)
│   ├── torch_model.py                 <- 2D Attention-augmented heuristic model (ChrestienHeuristicNet)
│   ├── confidence_aware_search.py     <- Test-time MC Dropout, PercentileTracker, Confidence-Aware A*
│   ├── ensemble_search.py             <- Deep Ensemble confidence-aware heuristic search engine
│   └── search_algorithms.py           <- Baseline A* and GBFS search implementations
│
├── benchmarks/                        <- Rigorous benchmark execution scripts
│   ├── run_rigorous_600s_benchmark.py <- 3-box in-distribution 600s benchmark runner
│   ├── run_5box_benchmark.py          <- 5-box OOD benchmark runner with periodic logging (every 20 runs)
│   ├── run_ensemble_benchmark.py      <- Deep Ensemble benchmark evaluation runner
│   └── generate_5box_dataset.py       <- Procedural 5-box Sokoban maze generator (gym-sokoban)
│
├── data/                              <- Benchmark problem datasets
│   ├── sokoban_3box_train.txt         <- Official NeurIPS 2023 training set (32,016 mazes)
│   ├── sokoban_3box_test.txt          <- Official NeurIPS 2023 test set (200 mazes)
│   └── sokoban_5box_test.txt          <- Procedurally generated 5-box test set (100 mazes)
│
├── results/                           <- Published benchmark evaluation logs and reports
│   ├── 3box_rigorous/                 <- Complete 3-box raw CSV, Markdown report, and console log
│   │   ├── rigorous_benchmark_results.csv
│   │   ├── rigorous_benchmark_report.md
│   │   └── rigorous_benchmark_run.log
│   └── 5box_ood/                      <- 5-box benchmark raw CSV and Markdown report
│       ├── sokoban_5box_benchmark_results.csv
│       └── sokoban_5box_benchmark_report.md
│
├── docs/                              <- Research documentation and theoretical proofs
│   └── confidence_aware_mc_dropout_theory.md
│
└── tools/                             <- Model conversion and training utilities
    ├── convert_finalSok3.py           <- Weight converter from author's TensorFlow checkpoint
    ├── train_deep_ensemble.py         <- Multi-seed 80% bootstrap bagging training pipeline
    └── train_baseline_lstar.py        <- Training pipeline for single ranking heuristic
```

---

## 7. Setup & Reproduction

### Prerequisites
- Python 3.10+ (tested on Python 3.14 on Windows 11)
- PyTorch 2.x with CUDA or Intel Arc XPU support

### Installation
```bash
git clone https://github.com/harshit-iitr/Confidence_aware_a_search.git
cd Confidence_aware_a_search
pip install -r requirements.txt
```

### Reproducing 3-Box Benchmark (In-Distribution)
```bash
python benchmarks/run_rigorous_600s_benchmark.py --timeout 600.0 --num_mazes 200
```

### Reproducing 5-Box Benchmark (Out-of-Distribution)
```bash
python benchmarks/run_5box_benchmark.py --timeout 600.0 --num_mazes 100 --log_interval 20
```

---

## 7. Citation & Acknowledgments
If you use this codebase or methodology in your research, please cite:
```bibtex
@inproceedings{chrestien2023optimize,
  title={Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal},
  author={Chrestien, Marin and Godet, Pierre and Kishimoto, Akihiro and Marinescu, Radu and Petit, Nicolas},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
