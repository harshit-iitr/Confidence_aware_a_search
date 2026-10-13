# 🚀 Guide for Running the 6x6 (35-Puzzle) Benchmark Tonight

This guide contains simple, foolproof instructions for cloning this repository and running the **$6\times 6$ (35-Puzzle) Sliding-Tile Benchmark** on your machine overnight.

---

## 📌 Task Overview

You are running the $6\times 6$ out-of-distribution (OOD) benchmark evaluating the pre-trained neural heuristic network from Chrestien et al. (*NeurIPS 2023*) across 3 methods:
1. **Paper Baseline A\* ($\mathcal{L}^*$):** The original deterministic paper heuristic.
2. **MC-Dropout Risk-Averse A\* ($\kappa = 0.50$):** Penalizes epistemic branch ambiguity ($f = g + \mu + \kappa \sigma$).
3. **MC-Dropout Confidence-Aware A\* ($\beta = 2.5$):** Dynamically gates between neural heuristic and classical Manhattan+Linear Conflict based on prediction reliability.

---

## 🛠️ Step 1: Clone the Repository & Setup Environment

Open your terminal (PowerShell, Bash, or Command Prompt):

```bash
# 1. Clone repository
git clone https://github.com/harshit-iitr/Confidence_aware_a_search.git
cd Confidence_aware_a_search

# 2. (Optional) Activate your Python virtual environment if you use one
# e.g.: conda activate myenv   OR   venv\Scripts\activate

# 3. Ensure PyTorch and basic requirements are installed
pip install torch numpy scipy
```

> **Note on Hardware:** The script automatically detects your GPU (**NVIDIA CUDA** or **Intel Arc XPU**). If no GPU is available, it automatically runs safely on CPU.

---

## ⚡ Step 2: Launch the Benchmark (Single Command)

Run the dedicated runner script:

```bash
python tools/run_strict_stile6_benchmark.py
```

### Useful Command Options (Optional):

| Flag | Default | Description | Example |
| :--- | :---: | :--- | :--- |
| `--timeout` | `360.0` | Timeout per board in seconds (6 minutes, matching paper) | `--timeout 180` |
| `--max_boards` | `200` | Total boards to evaluate (from `stile6.txt`) | `--max_boards 50` |
| `--start_idx` | `0` | Starting board index (0 to 199) | `--start_idx 0` |
| `--end_idx` | `200` | Ending board index | `--end_idx 100` |
| `--device` | Auto | Force device: `cuda`, `xpu`, or `cpu` | `--device cuda` |

#### Example: Overnight Full Run (Recommended)
```bash
python tools/run_strict_stile6_benchmark.py --timeout 360 --max_boards 200
```

#### Example: Quick Test Run (First 5 boards, 60s timeout)
```bash
python tools/run_strict_stile6_benchmark.py --timeout 60 --max_boards 5
```

---

## 📊 Step 3: What to Expect During the Run

1. **Live Console Output:** You will see real-time updates as each board finishes:
   ```text
   --- [Board #  0/200 | Init MD+LC: 72.0] ---
     [paper_baseline_lstar] SOLVED (412 exp, len 68) in 1.45s
     [mc_dropout_risk_averse] SOLVED (298 exp, len 68) in 1.12s
     [mc_dropout_conf_aware] SOLVED (210 exp, len 66) in 0.88s
   ```
2. **Crash-Proof & Resume-Safe:** Every single board is immediately written to disk:
   - Raw data: [`results/sliding_tile_6x6_strict_paper_benchmark.csv`](file:///d:/IITR/DAC-203_project/results/sliding_tile_6x6_strict_paper_benchmark.csv)
   - Live Markdown summary: [`results/sliding_tile_6x6_strict_paper_report.md`](file:///d:/IITR/DAC-203_project/results/sliding_tile_6x6_strict_paper_report.md)
   > *If the script is stopped or interrupted for any reason, simply rerun the exact same command. It reads the CSV and automatically skips all already-completed boards!*

---

## 📤 Step 4: After Completion (Pushing or Sharing Results)

Once finished (or in the morning), commit and push the generated results:

```bash
git add results/sliding_tile_6x6_strict_paper_benchmark.csv results/sliding_tile_6x6_strict_paper_report.md
git commit -m "Add 6x6 sliding-tile benchmark results"
git push origin main
```

*(Or simply email/send the `results/sliding_tile_6x6_strict_paper_benchmark.csv` file back!)*
