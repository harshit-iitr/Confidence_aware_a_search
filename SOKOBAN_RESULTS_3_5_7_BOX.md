# Sokoban results — 3-box, 5-box and 7-box

Reporting cutoff: **maps 1–82 for seven-box Sokoban**, chosen at the user's request on 1 November 2026. No later maps are included in these tables. The seven-box study is a partial 82/100 evaluation, not a completed 100-map benchmark.

**Data status:** 3/5-box values come from the original repository's full benchmark CSVs, excluding recent conversion reruns and small exploratory pilots. Seven-box values combine downloaded raw results for maps 1–45 with completed live-log records for maps 46–82. The active Kaggle job had reached map 83 when sampled; its final archive was not yet available. All 738 A* attempts through map 82 are represented. The live log's protocol fingerprint and per-attempt seeds match the archived runs. Saved plans for the latest 37 maps have not yet been downloaded and independently replayed locally; solution costs are therefore omitted from the seven-box table. This file is a completed reporting snapshot, not a claim that the active cloud session was stopped or archived.

**Reading the tables:** solve rates include all evaluated maps. Mean, median and P90 expansions, and mean solution cost, use each method's solved maps. Different solved sets make these expansion averages unsuitable as paired comparisons; paired findings are given below. Method names follow the original repo, including its historical “Paper Learned” label; this is not a verified paper-reproduction claim. GBFS is retained only in historical 3/5-box results; seven-box uses A* only.

**Setup keys:** A = 600 seconds / 100,000 expansions; B = 360 seconds / 1,000,000 expansions; D = historical 600 seconds, with configured ceilings of 15,000 neural / 150,000 classical expansions. Historical loop behavior can overshoot. XPU = Intel Arc 130V. RTX 3050 = NVIDIA RTX 3050 Laptop GPU. Seven-box uses Tesla T4 (classical search remains on CPU) and budget B for every method. These are separate experiment groups, not one uniform hardware/budget comparison across domains.

## 3-box Sokoban — IID, 200 maps

| Method | Setup | Solved | Mean exp. | Median exp. | P90 exp. | Mean cost |
|---|---|---:|---:|---:|---:|---:|
| Classical A* (Manhattan) | CPU / A | 195/200 (97.5%) | 11,133.3 | 3,370.0 | 36,299.8 | 26.0 |
| Paper Learned A* | XPU / A | 200/200 (100.0%) | 1,262.9 | 224.5 | 3,694.1 | 26.6 |
| Paper Learned GBFS | XPU / A | 200/200 (100.0%) | 944.4 | 43.0 | 1,689.8 | 31.3 |
| Fixed 50/50 Hybrid A* | XPU / A | 200/200 (100.0%) | 762.2 | 155.5 | 1,719.3 | 29.0 |
| Fixed 50/50 Hybrid GBFS | XPU / A | 200/200 (100.0%) | 1,561.0 | 93.0 | 4,959.9 | 33.3 |
| MC Confidence-Aware A* — paper model | XPU / A | 200/200 (100.0%) | 596.0 | 92.0 | 1,393.5 | 28.4 |
| MC Confidence-Aware GBFS — paper model | XPU / A | 200/200 (100.0%) | 1,241.2 | 68.5 | 2,168.8 | 32.6 |
| Single Scratch Model #0 A* | XPU / B | 200/200 (100.0%) | 1,222.3 | 121.0 | 3,446.8 | 27.1 |
| Normal 5-Model Ensemble A* | XPU / B | 200/200 (100.0%) | 377.0 | 89.0 | 757.4 | 29.6 |
| Confidence-Aware Ensemble A* (conf_20) | XPU / B | 200/200 (100.0%) | 377.9 | 90.0 | 708.1 | 29.4 |

## 5-box Sokoban — OOD, 100 maps

| Method | Setup | Solved | Mean exp. | Median exp. | P90 exp. | Mean cost |
|---|---|---:|---:|---:|---:|---:|
| Classical A* (Manhattan) | CPU / D | 56/100 (56.0%) | 41,546.2 | 26,794.0 | 104,652.0 | 31.7 |
| Paper Learned A* | XPU / D | 71/100 (71.0%) | 2,833.8 | 1,384.0 | 8,354.0 | 33.5 |
| Paper Learned GBFS | XPU / D | 70/100 (70.0%) | 2,516.1 | 346.0 | 8,529.8 | 45.9 |
| MC Confidence-Aware A* — paper model | XPU / D | 90/100 (90.0%) | 1,970.5 | 567.5 | 6,921.5 | 38.3 |
| MC Confidence-Aware GBFS — paper model | XPU / D | 66/100 (66.0%) | 1,957.4 | 305.5 | 7,394.5 | 44.5 |
| Single Scratch Model #0 A* | XPU / B | 93/100 (93.0%) | 8,740.4 | 1,436.0 | 25,827.2 | 37.5 |
| Normal 5-Model Ensemble A* | XPU / B | 95/100 (95.0%) | 1,665.1 | 591.0 | 5,672.2 | 41.1 |
| Confidence-Aware Ensemble A* (conf_20) | XPU / B | 95/100 (95.0%) | 1,614.9 | 549.0 | 4,601.8 | 41.0 |
| MC-Dropout on Scratch Model #0 | XPU / B | 77/100 (77.0%) | 7,906.9 | 3,042.0 | 20,360.2 | 36.4 |
| Calibrated Confidence-Aware Ensemble A* (conf_50) | XPU / B | 94/100 (94.0%) | 1,553.5 | 513.0 | 4,697.7 | 40.8 |
| Normal 5-Model Ensemble A* | RTX 3050 / B | 93/100 (93.0%) | 1,407.6 | 559.0 | 4,024.2 | 40.7 |
| Risk-Averse Ensemble A* (kappa=1) | RTX 3050 / B | 94/100 (94.0%) | 1,354.8 | 393.0 | 4,122.2 | 41.6 |
| Calibrated Confidence-Aware Ensemble A* (conf_50) | RTX 3050 / B | 92/100 (92.0%) | 1,391.6 | 501.5 | 4,065.6 | 40.5 |

## 7-box Sokoban — OOD, 82 completed maps

Same nine A* configurations, frozen maps 1–82, Tesla T4, 360 seconds / 1,000,000 expansions. Latest 37 maps are a live-log snapshot pending raw artifact export.

| Method | Solved | Mean exp. | Median exp. | P90 exp. |
|---|---:|---:|---:|---:|
| Classical A* (Manhattan) | 40/82 (48.8%) | 294,972.6 | 156,048.0 | 848,286.1 |
| Paper Learned A* | 61/82 (74.4%) | 19,956.2 | 5,782.0 | 62,134.0 |
| MC Confidence-Aware A* — paper model | 63/82 (76.8%) | 9,243.6 | 1,246.0 | 17,463.4 |
| Single Scratch Model #0 A* | 65/82 (79.3%) | 13,141.4 | 2,289.0 | 38,269.4 |
| MC-Dropout on Scratch Model #0 | 53/82 (64.6%) | 14,561.3 | 7,295.0 | 35,464.4 |
| Normal 5-Model Ensemble A* | 60/82 (73.2%) | 3,079.2 | 869.0 | 12,023.1 |
| Confidence-Aware Ensemble A* (conf_20) | 62/82 (75.6%) | 3,111.1 | 908.0 | 8,960.9 |
| Calibrated Confidence-Aware Ensemble A* (conf_50) | 61/82 (74.4%) | 3,238.1 | 871.0 | 7,887.0 |
| Risk-Averse Ensemble A* (kappa=1) | 62/82 (75.6%) | 2,883.8 | 735.5 | 11,059.6 |

### Seven-box paired comparisons

- **MC Confidence-Aware A* — paper model vs Paper Learned A*:** 61 mutually solved maps; 54 wins / 0 ties / 7 losses; 63.89% mean expansion reduction (negative means more expansions).
- **Risk-Averse Ensemble A* (kappa=1) vs Normal 5-Model Ensemble A*:** 60 mutually solved maps; 50 wins / 0 ties / 10 losses; 29.11% mean expansion reduction (negative means more expansions).
- **Confidence-Aware Ensemble A* (conf_20) vs Normal 5-Model Ensemble A*:** 60 mutually solved maps; 25 wins / 0 ties / 35 losses; 15.08% mean expansion reduction (negative means more expansions).
- **Calibrated Confidence-Aware Ensemble A* (conf_50) vs Normal 5-Model Ensemble A*:** 60 mutually solved maps; 23 wins / 1 ties / 36 losses; 5.83% mean expansion reduction (negative means more expansions).
- **MC-Dropout on Scratch Model #0 vs Single Scratch Model #0 A*:** 53 mutually solved maps; 3 wins / 0 ties / 50 losses; -259.32% mean expansion reduction (negative means more expansions).

### Reporting closeout

The requested report stops at 82 maps. Maps 83–100 are outside this snapshot. No further Kaggle sessions were launched for this report. The existing session was still running, so its safe export remains a separate outstanding step; it was not cancelled while its archive was unavailable. All historical and live-log evidence is preserved. Final solution-length comparisons require the latest raw artifact.

### Limitation: MC dropout on scratch M0

MC confidence-aware search is model-dependent. On five-box problems, scratch M0 solves 93/100 while its MC variant solves 77/100. In this seven-box snapshot, those counts are 65/82 and 53/82. On the 53 mutually solved seven-box maps, scratch MC uses approximately 3.6 times as many mean expansions. This cannot be explained by extra forward-pass time alone.

Possible explanations include sensitivity to inference-time dropout, degraded prediction rankings, unreliable disagreement, and interaction with classical blending. These are hypotheses, not established causes. The comparison changes dropout sampling, percentile scaling, and confidence blending together; isolating their effects requires separate ablations. No claim is made that disagreement predicts physical distance error.

### Sources

- `results/3box_rigorous/rigorous_benchmark_results.csv`
- `results/ensemble_3box_benchmark_results.csv`
- `results/5box_ood/sokoban_5box_benchmark_results.csv`
- `results/ensemble_5box_benchmark_results.csv`
- `results/calibrated_5box_experiment_results.csv`
- `results/riskaverse_5box_benchmark_results.csv`
- `results/sokoban7_original/maps001_005/`
- `results/sokoban7_original/maps006_010/`
- `results/sokoban7_original/maps011_100_session1/`
- `results/sokoban7_original/snapshot_82maps/live_events.json` (completed END records only, through map 82)
- `COMPILED_SOKOBAN_RESULTS.md` retains the earlier exploratory pilots separately.
