# 7-box original-repository protocol

Complete across all 11 configurations: **5/100 maps**. Unbalanced records excluded from comparison: 0.

Budget: 360.0 seconds / 1,000,000 expansions per attempt. Hardware: CUDA (Tesla T4). Kind: 7-box-test.

Expansion/cost statistics use each method's solved subset; all-run time includes failures. Use the paired comparisons below to compare expansions on identical solved maps.

| Method | Solved | Mean exp. | Median exp. | P90 exp. | Mean cost | Mean time, all (s) |
|---|---:|---:|---:|---:|---:|---:|
| Classical A* (Manhattan) | 3/5 | 126,227.00 | 77,764.00 | 227,343.20 | 35.67 | 28.51 |
| Paper Learned A* (original repository checkpoint) | 4/5 | 10,879.75 | 3,078.00 | 26,895.30 | 36.50 | 93.00 |
| Paper Learned GBFS (original repository checkpoint) | 4/5 | 40,236.75 | 1,606.50 | 111,265.20 | 46.00 | 140.82 |
| MC Confidence-Aware Hybrid A* (paper model) | 4/5 | 1,388.25 | 1,093.00 | 2,393.50 | 39.00 | 75.55 |
| MC Confidence-Aware Hybrid GBFS (paper model) | 3/5 | 33,183.33 | 664.00 | 79,136.00 | 54.00 | 207.51 |
| Single Scratch Model #0 A* | 4/5 | 2,847.25 | 2,800.00 | 5,131.50 | 37.75 | 77.72 |
| MC-Dropout on Scratch Model #0 | 4/5 | 13,551.00 | 7,978.50 | 28,217.20 | 41.25 | 106.97 |
| Normal 5-Model Ensemble A* | 4/5 | 627.75 | 603.50 | 965.00 | 44.00 | 78.21 |
| Confidence-Aware Ensemble A* (lambda_min=0.20) | 4/5 | 589.75 | 426.00 | 1,083.90 | 40.75 | 77.81 |
| Calibrated Confidence-Aware Ensemble A* (lambda_min=0.50) | 4/5 | 674.50 | 600.50 | 1,092.50 | 43.75 | 78.73 |
| Risk-Averse Ensemble A* (kappa=1.0) | 4/5 | 576.75 | 570.00 | 990.90 | 43.75 | 77.78 |

Paired mean expansion reduction (positive means fewer expansions):

- paper_mc_astar vs paper_astar: N=4; 3W/0T/1L; 87.24%.
- paper_mc_gbfs vs paper_gbfs: N=3; 0W/0T/3L; -2,916.67%.
- ensemble_risk vs ensemble_mean: N=4; 3W/0T/1L; 8.12%.
- ensemble_conf_020 vs ensemble_mean: N=4; 1W/0T/3L; 6.05%.
- ensemble_conf_050 vs ensemble_mean: N=4; 2W/0T/2L; -7.45%.
