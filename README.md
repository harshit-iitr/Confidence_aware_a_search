# Confidence-Aware Rank-Blended Heuristic Search

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(XPU%2FCUDA)-EE4C2C.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Intel%20Arc%20GPU%20(XPU)%20%2F%20CUDA-0071C5.svg)](https://www.intel.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An adaptive, confidence-aware heuristic search framework that dynamically estimates the reliability of learned neural ranking heuristics online and modulates the blend between learned and classical admissible heuristics.

This project builds upon and extends the foundation established in:
> **Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal**  
> *Chrestien et al., Advances in Neural Information Processing Systems (NeurIPS 2023).*

---

## 1. Motivation & Core Concept

Standard learned heuristics trained via pairwise ranking losses ($\mathcal{L}^*$) excel at ordering states along promising trajectories, but exhibit two critical vulnerabilities:
1. **Uncalibrated Scales in A\*:** Ranking models output arbitrary scalars without physical cost calibration, causing severe scale distortions when directly computing $f(s) = g(s) + h_\theta(s)$.
2. **Catastrophic Deadlocks on Ambiguous / OOD States:** When traversing out-of-distribution (OOD) or unfamiliar maze topologies, deterministic neural heuristics make overconfident erroneous decisions, trapping search in extensive deadlocks.

**Our Proposed Framework:** We introduce a **Confidence-Aware Rank-Blended A\*** search:
* **Test-Time Monte Carlo Dropout:** Samples $M$ stochastic predictions in a single parallel tensor pass to quantify epistemic uncertainty online without training multiple models from scratch.
* **Global Percentile Tracking ($\mathcal{H}_m, \mathcal{H}_{class}$):** Maps both the uncalibrated neural heuristics and the classical heuristic into a normalized rank-percentile space $[0, 1]$ in $O(\log V)$ per expansion.
* **Confidence Gating Function ($\lambda(s)$):** Computes rank variance $\sigma^2_{rank}(s) \in [0, 0.25]$ and modulates confidence:
  $$\lambda(s) = \lambda_{min} + (1 - \lambda_{min}) \cdot \max\big(0,\ 1 - 4\sigma^2_{rank}(s)\big)$$
* **Rank-Space Convex Blend:** Converts the convex combination $p_{blend}(s)$ back to cost space via a single static scale constant $C$:
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

## 3. Empirical Results (Sokoban 10x10 Benchmark)

Comparison on benchmark Sokoban instances comparing Classical A\*, Paper Baseline Learned A\*, and our Confidence-Aware Hybrid A\*:

| Method | Solve Rate | Mean Node Expansions | Median Expansions | Mean Path Cost | Deadlock Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Classical A\* (Hungarian Manhattan)** | 75.0% | 2,418.3 | 2,653.0 | **20.33** | Zero deadlocks, high exploration |
| **Paper Learned A\* (NeurIPS 2023 Baseline)** | 75.0% | 130.3 | 129.5 | 21.17 | Trapped in deadlocks on Maps 4 & 7 |
| **Confidence-Aware Hybrid A\* (Ours)** | **100.0%** | **78.1** | **65.5** | 25.00 | **Zero deadlocks (Safely navigated)** |

* **Expansion Reduction:** **~40% to 50% fewer node expansions** than the paper's neural baseline, and **96.8% fewer expansions** than classical A\*.
* **Deadlock Recovery:** Successfully solved the exact benchmark maps where the paper's baseline timed out due to overconfident deadlocks.

---

## 4. Repository Structure

```
├── confidence_aware_search.py       # Core search engine (PercentileTracker, MC Dropout, Hybrid A*)
├── torch_model.py                   # PyTorch Neural Heuristic (Conv2D + 2D Attention + Positional Encoding)
├── classical_heuristics.py          # Classical admissible Sokoban heuristic (Hungarian bipartite matching)
├── sokoban_env.py                   # Sokoban grid environment, state tensor encoders & transition logic
├── sanity_check_step1.py            # Step 1 Research Integrity & Sanity Audit Suite
├── test_confidence_aware_small.py   # Comparative verification test across sample mazes
├── run_deep_validation_test.py      # In-distribution vs. OOD uncertainty calibration suite
├── eval_baseline.py                 # Benchmarking script for classical vs. paper learned heuristics
├── train_baseline_lstar.py          # High-performance PyTorch training pipeline on XPU/CUDA
├── finalSok3_pytorch.pt             # Pre-trained baseline checkpoint
├── confidence_aware_mc_dropout_theory.md # Formal theoretical manuscript & proofs
└── requirements.txt                 # Python dependencies
```

---

## 5. Getting Started

### Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/harshit-iitr/confidence-aware-heuristic-search.git
cd confidence-aware-heuristic-search
pip install -r requirements.txt
```

### Running the Research Integrity Audit (Step 1)
Verify zero data leakage, mathematical invariants, and physical action replays:
```bash
python sanity_check_step1.py
```

### Running the Deep Uncertainty Validation
Inspect the in-distribution vs. out-of-distribution uncertainty gating behavior:
```bash
python run_deep_validation_test.py
```

### Running Comparative Small Tests
Run a quick comparative test across sample benchmark maps:
```bash
python test_confidence_aware_small.py
```

---

## 6. Citation & References

* Chrestien et al., *Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal*, NeurIPS 2023.
* Yonetani et al., *Path Planning using Neural A\* Search*, ICML 2021.
* Gal & Ghahramani, *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*, ICML 2016.
