# Seven-box transfer: original repository protocol

This evaluates the original five-box implementations on 100 frozen seven-box
Sokoban puzzles on 10x10 grids. It reuses `finalSok3_pytorch.pt` and ensemble
members 0 through 4. There is no training or checkpoint conversion. The existing
search engines are imported directly and remain unchanged. "Paper Learned" is
the original repository's method name, not a verified reproduction claim.

## Uniform evaluation budget

Every method gets **360 seconds and 1,000,000 expansions per puzzle**. This
deliberately changes the historical five-box baseline suite's stopping budgets.
Search loops check budgets at their original boundaries, so a final operation can
slightly overrun. Their existing expansion counters and closed-list behavior are
retained. Search time excludes heuristic-wrapper construction; `wall_time_sec`
additionally records that construction. Failed searches retain their time and
expansion counts. No symbolic deadlock pruning is enabled for any method.

All methods use the same frozen boards and GPU. CUDA and XPU repetitions are not
different algorithms. The 13 historical five-box configurations become 11:

1. Classical Manhattan A* (CPU search within the same session).
2. Original paper-model deterministic A*.
3. Original paper-model deterministic GBFS.
4. Paper-model MC confidence-aware A*.
5. Paper-model MC confidence-aware GBFS.
6. Single trained member M0 A*.
7. M0 MC confidence-aware A*.
8. Five-model percentile-mean A* (fixed lambda=1).
9. Confidence-aware ensemble A* (lambda_min=0.20).
10. Calibrated confidence-aware ensemble A* (lambda_min=0.50).
11. Risk-averse ensemble A* (kappa=1.0).

MC settings: 5 samples, dropout 0.10, lambda_min 0.20, C=50. Ensemble settings:
beta=2.5, std gating, C=50. Risk penalty follows the existing implementation,
including clipping the penalized percentile to 1. Uncertainty is computed in
per-member percentile space, not from raw cross-model offsets. Fresh trackers
are created for each method/map. Independent seeds derive from base seed, map ID,
and method ID, so resuming and splitting jobs do not alter stochastic streams.
Hardware-dependent floating-point behavior and timeouts can still vary.

## Frozen test set

`data/sokoban_7box_original_test.txt` contains 100 unique puzzles. Its provenance
records the generator version, seed, attempts, parameters, and dataset SHA256.
Generation follows `benchmarks/generate_5box_dataset.py`: `generate_room` with
10x10, num_steps=30, tries=20, seed=42, changing num_boxes to 7. Both NumPy and
Python RNGs are seeded explicitly. Selection uses dimensions, tile counts,
closed boundaries, and uniqueness only. No puzzles are selected by solver
performance. Seven boxes are not proof of matched published benchmark difficulty;
num_steps=30 is not the true solution length.

To regenerate in a **different output path** (never replace the frozen set):

```bash
pip install gym-sokoban==0.0.6
python tools/generate_sokoban7_original.py --output output/independent_set.txt
```

## Run and resume

```bash
python benchmarks/run_sokoban7_benchmark.py run --device cuda:0 --start-map 1 --end-map 5 --output output/7box/maps001_005
```

Repeat with disjoint ranges 6–10, 11–15, etc. in separate output directories.
The default session wall allowance is 7.5 hours. The runner stops before starting
an attempt when fewer than 375 seconds remain. Repeating the command with the
same output directory resumes unfinished attempts. Preserve `manifest.json` and
`results.jsonl` together when moving results between sessions. Resume requires
the same code, models, dataset, seed, and recorded hardware/runtime. A stale
`.run.lock` may be removed only after confirming its process is no longer alive.

There are 1,100 attempts, with a maximum **110 search-hours**. Five maps take at
most 5.5 search-hours, plus setup and I/O. One eight-hour Kaggle session is not
enough for a guaranteed full comparison. Measure the first shard for a forecast.

Merge disjoint shards from the same recorded environment:

```bash
python benchmarks/run_sokoban7_benchmark.py report output/7box/maps001_005 output/7box/maps006_010 --output output/7box/combined
```

The Markdown, CSV, and JSON summaries use only maps completed by all 11 methods.
They show completion counts explicitly. Expansions and cost use solved maps;
runtime including failures is also reported. Paired comparisons use mutually
solved maps. Duplicate or incompatible shards are refused. All successful plans
are saved and independently replayed, including on resume.

## Validation and Kaggle preparation

```bash
python -m unittest discover -s tests -p test_sokoban7_original.py -v
python benchmarks/run_sokoban7_benchmark.py run --smoke-test --device cpu --output output/7box_smoke
python tools/package_sokoban7_original.py
```

The smoke test uses a one-box one-push board, a 10-second/100-expansion budget,
and all 11 configurations with the actual checkpoints. It is labelled validation
data and cannot merge with the experiment. The allowlisted zip contains source,
the six original checkpoints, frozen data, and license; credentials are excluded.
`kaggle_sokoban7_original.ipynb` extracts and verifies this bundle, runs one
five-map shard on CUDA, and archives results under `/kaggle/working`.

No credentials, upload, or GPU launch are required to prepare this code.

### Generate and freeze on Kaggle before evaluation

If local generation is unfinished, build with
`python tools/package_sokoban7_original.py --generate-in-cloud`. This explicit
mode packages the original checkpoints and generator without a test set.
`tools/run_sokoban7_cloud.py` then generates **all 100 maps before the first
search**, preserving the same generation settings. Failed-generation
`RuntimeWarning` exceptions are retried just like the original five-box script.
Accepted maps and both RNG states are checkpointed after every attempt; a
generation interruption can be resumed in the same environment without losing
previously accepted maps. The final dataset and provenance are archived with
the search results. Later shards must reuse that frozen dataset, not regenerate
independent samples. Generation time counts against the 7.5-hour cloud allowance.

The launcher accepts an explicit credentials-file path and uses its values only
in memory. It uploads a private allowlisted dataset and launches one private
eight-hour GPU kernel. It does not upload the credentials file. Never commit a
key or put it in the bundle.
