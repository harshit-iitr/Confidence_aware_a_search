# 7-box original-repository protocol

Complete across all 11 configurations: **0/100 maps**. Unbalanced records excluded from comparison: 45.

Budget: 360.0 seconds / 1,000,000 expansions per attempt. Hardware: CUDA (Tesla T4). Kind: 7-box-test.

Expansion/cost statistics use each method's solved subset; all-run time includes failures. Use the paired comparisons below to compare expansions on identical solved maps.

| Method | Solved | Mean exp. | Median exp. | P90 exp. | Mean cost | Mean time, all (s) |
|---|---:|---:|---:|---:|---:|---:|
| Classical A* (Manhattan) | 0/0 | — | — | — | — | — |
| Paper Learned A* (original repository checkpoint) | 0/0 | — | — | — | — | — |
| Paper Learned GBFS (original repository checkpoint) | 0/0 | — | — | — | — | — |
| MC Confidence-Aware Hybrid A* (paper model) | 0/0 | — | — | — | — | — |
| MC Confidence-Aware Hybrid GBFS (paper model) | 0/0 | — | — | — | — | — |
| Single Scratch Model #0 A* | 0/0 | — | — | — | — | — |
| MC-Dropout on Scratch Model #0 | 0/0 | — | — | — | — | — |
| Normal 5-Model Ensemble A* | 0/0 | — | — | — | — | — |
| Confidence-Aware Ensemble A* (lambda_min=0.20) | 0/0 | — | — | — | — | — |
| Calibrated Confidence-Aware Ensemble A* (lambda_min=0.50) | 0/0 | — | — | — | — | — |
| Risk-Averse Ensemble A* (kappa=1.0) | 0/0 | — | — | — | — | — |

Paired mean expansion reduction (positive means fewer expansions):

- paper_mc_astar vs paper_astar: N=0; 0W/0T/0L; —%.
- paper_mc_gbfs vs paper_gbfs: N=0; 0W/0T/0L; —%.
- ensemble_risk vs ensemble_mean: N=0; 0W/0T/0L; —%.
- ensemble_conf_020 vs ensemble_mean: N=0; 0W/0T/0L; —%.
- ensemble_conf_050 vs ensemble_mean: N=0; 0W/0T/0L; —%.
