# Original-repository Sokoban results

Scope: GitHub main commit `725d90187512da7701d36e59786797dab6d310fb` only. Recent conversion reruns and seven-box experiments are excluded. Method names follow the original repository, including its "Paper Learned" terminology; this compilation does not establish checkpoint provenance or reproduction fidelity.

Statistics are recalculated from original CSVs. Expansions, cost, time, and lambda are averaged over solved maps only; median and P90 also use solved maps. Solve rates use all recorded attempts for that row. Different solved subsets, hardware, and budgets must remain visible. Duplicate CSV copies at repository root are counted once. Saved verified flags are true for all solved non-legacy records; the legacy CSV has no verification field.

Budget keys: **A:** 600 s / 100,000 expansions. **B:** 360 s / 1,000,000 expansions. **C:** early pilot, 10 s / 15,000 expansions. **D:** 600 s; configured neural ceiling 15,000 and classical ceiling 150,000 (historical stopping behavior may overshoot). **E:** partial selected-hard-map pilot with unequal recorded cohorts (8 versus 7 attempts); hardware is not recorded in its CSV. These rows are not a completed ten-map experiment. A–D run settings come from the original benchmark reports and scripts.

XPU denotes Intel Arc 130V; RTX 3050 denotes NVIDIA RTX 3050 Laptop GPU. Classical methods use CPU search in those suites. Paper-model rows use the repository's existing finalSok3 PyTorch port; trained-model rows use ensemble member M0 or all five ensemble members. Confidence-aware ensemble and calibrated ensemble rows are retained separately.

## 3-box Sokoban — IID

| Method / configuration | Evaluation | Solved | Mean expansions | Median expansions | P90 expansions | Mean cost | Mean time (s) | Mean lambda |
|---|---|---|---|---|---|---|---|---|
| Classical A* (Manhattan) | 200 maps; XPU; A | 195/200 (97.5%) | 11,133.3 | 3,370.0 | 36,299.8 | 26.0 | 0.232 | — |
| Paper Learned A* | 200 maps; XPU; A | 200/200 (100%) | 1,262.9 | 224.5 | 3,694.1 | 26.6 | 4.517 | — |
| Paper Learned GBFS | 200 maps; XPU; A | 200/200 (100%) | 944.4 | 43.0 | 1,689.8 | 31.3 | 2.918 | — |
| Fixed 50/50 Hybrid A* | 200 maps; XPU; A | 200/200 (100%) | 762.2 | 155.5 | 1,719.3 | 29.0 | 3.655 | 0.500 |
| Fixed 50/50 Hybrid GBFS | 200 maps; XPU; A | 200/200 (100%) | 1,561.0 | 93.0 | 4,959.9 | 33.3 | 7.287 | 0.500 |
| MC-Dropout Confidence-Aware Hybrid A* (paper model) | 200 maps; XPU; A | 200/200 (100%) | 596.0 | 92.0 | 1,393.5 | 28.4 | 3.042 | 0.879 |
| MC-Dropout Confidence-Aware Hybrid GBFS (paper model) | 200 maps; XPU; A | 200/200 (100%) | 1,241.2 | 68.5 | 2,168.8 | 32.6 | 5.805 | 0.876 |
| Single Scratch Model #0 A* | 200 maps; XPU; B | 200/200 (100%) | 1,222.3 | 121.0 | 3,446.8 | 27.1 | 3.864 | 1.000 |
| Normal 5-Model Ensemble A* | 200 maps; XPU; B | 200/200 (100%) | 377.0 | 89.0 | 757.4 | 29.6 | 6.046 | 1.000 |
| Confidence-Aware 5-Model Ensemble A* | 200 maps; XPU; B | 200/200 (100%) | 377.9 | 90.0 | 708.1 | 29.4 | 5.929 | 0.979 |
| Classical A* (Manhattan) | 50-map early pilot; XPU; C | 43/50 (86%) | 3,777.2 | 2,312.0 | 9,449.4 | 22.1 | 0.170 | — |
| Paper Learned A* | 50-map early pilot; XPU; C | 38/50 (76%) | 188.2 | 127.0 | 393.4 | 21.0 | 2.321 | — |
| Paper Learned GBFS | 50-map early pilot; XPU; C | 46/50 (92%) | 127.0 | 28.0 | 350.5 | 25.1 | 1.332 | — |
| Fixed 50/50 Hybrid A* | 50-map early pilot; XPU; C | 43/50 (86%) | 173.8 | 106.0 | 443.2 | 24.5 | 2.313 | 0.500 |
| MC-Dropout Confidence-Aware Hybrid A* | 50-map early pilot; XPU; C | 47/50 (94%) | 125.6 | 70.0 | 260.6 | 25.0 | 2.068 | 0.885 |

## 5-box Sokoban — OOD

| Method / configuration | Evaluation | Solved | Mean expansions | Median expansions | P90 expansions | Mean cost | Mean time (s) | Mean lambda |
|---|---|---|---|---|---|---|---|---|
| Classical A* (Manhattan) | 100 maps; XPU; D | 56/100 (56%) | 41,546.2 | 26,794.0 | 104,652.0 | 31.7 | 1.129 | — |
| Paper Learned A* | 100 maps; XPU; D | 71/100 (71%) | 2,833.8 | 1,384.0 | 8,354.0 | 33.5 | 12.719 | — |
| Paper Learned GBFS | 100 maps; XPU; D | 70/100 (70%) | 2,516.1 | 346.0 | 8,529.8 | 45.9 | 9.465 | — |
| MC-Dropout Confidence-Aware Hybrid A* (paper model) | 100 maps; XPU; D | 90/100 (90%) | 1,970.5 | 567.5 | 6,921.5 | 38.3 | 10.411 | 0.887 |
| MC-Dropout Confidence-Aware Hybrid GBFS (paper model) | 100 maps; XPU; D | 66/100 (66%) | 1,957.4 | 305.5 | 7,394.5 | 44.5 | 11.276 | 0.883 |
| Single Scratch Model #0 A* | 100 maps; XPU; B | 93/100 (93%) | 8,740.4 | 1,436.0 | 25,827.2 | 37.5 | 28.657 | 1.000 |
| Normal 5-Model Ensemble A* | 100 maps; XPU; B | 95/100 (95%) | 1,665.1 | 591.0 | 5,672.2 | 41.1 | 27.176 | 1.000 |
| Confidence-Aware 5-Model Ensemble A* | 100 maps; XPU; B | 95/100 (95%) | 1,614.9 | 549.0 | 4,601.8 | 41.0 | 26.964 | 0.982 |
| MC-Dropout on Scratch Model #0 | 100 maps; XPU; B | 77/100 (77%) | 7,906.9 | 3,042.0 | 20,360.2 | 36.4 | 39.908 | 0.803 |
| Calibrated Confidence-Aware Ensemble A* | 100 maps; XPU; B | 94/100 (94%) | 1,553.5 | 513.0 | 4,697.7 | 40.8 | 24.605 | 0.930 |
| Normal 5-Model Ensemble A* | 100 maps; RTX 3050; B | 93/100 (93%) | 1,407.6 | 559.0 | 4,024.2 | 40.7 | 42.952 | 1.000 |
| Risk-Averse Deep Ensemble A* (kappa=1.0) | 100 maps; RTX 3050; B | 94/100 (94%) | 1,354.8 | 393.0 | 4,122.2 | 41.6 | 40.674 | 1.000 |
| Calibrated Confidence-Aware Ensemble A* (beta=2.5) | 100 maps; RTX 3050; B | 92/100 (92%) | 1,391.6 | 501.5 | 4,065.6 | 40.5 | 42.016 | 0.930 |
| Earlier Percentile Risk-Averse A* (kappa=1.0) | Selected hard-map pilot; partial; E | 7/8 (87.5%) | 3,615.1 | 3,096.0 | 7,020.6 | 53.0 | 75.971 | — |
| Goal-Anchored Risk-Averse A* (kappa=1.0, scale=1.5) | Selected hard-map pilot; partial; E | 6/7 (85.7143%) | 8,813.0 | 8,472.0 | 14,786.0 | 51.3 | 167.806 | — |

Sources (relative to this checkout):

- `results/3box_rigorous/rigorous_benchmark_results.csv`
- `results/ensemble_3box_benchmark_results.csv`
- `legacy/benchmark_results.csv`
- `results/5box_ood/sokoban_5box_benchmark_results.csv`
- `results/ensemble_5box_benchmark_results.csv`
- `results/calibrated_5box_experiment_results.csv`
- `results/riskaverse_5box_benchmark_results.csv`
- `results/hard10_goal_anchored_vs_riskaverse.csv`

The heuristic-error calibration experiment is a state-level analysis, not a search algorithm benchmark, so it does not form a solve-rate row. No unrun combinations are filled in.
