# Guide: Training the Sliding-Tile 5-Model Deep Ensemble on Kaggle (Tesla T4 GPU)

This guide provides step-by-step instructions to train the **5-Model Deep Ensemble** for the Sliding-Tile ($5\times 5$ 24-Puzzle) domain on **Kaggle's free Tesla T4 GPU**.

---

## 1. Quick Start in 4 Clicks on Kaggle

1. Go to [kaggle.com](https://www.kaggle.com/) and click **+ Create** $\rightarrow$ **New Notebook**.
2. In the right-hand **Notebook Settings** sidebar:
   - **Accelerator:** Select **GPU T4 x1** (free 30 hours/week).
   - **Internet:** Make sure **Internet on** is toggled **ON** (required to clone GitHub).
3. In the first notebook cell, paste and run:
   ```bash
   !git clone https://github.com/harshit-iitr/Confidence_aware_a_search.git
   %cd Confidence_aware_a_search
   ```
4. In the second cell, start the training pipeline:
   ```bash
   !python -u tools/train_sliding_tile_kaggle.py \
       --model_id all \
       --steps 20000 \
       --lr 3e-4 \
       --log_interval 500 \
       --val_interval 1000 \
       --zip_output /kaggle/working/slidingtile_checkpoints.zip
   ```
5. Once completed, download the zipped checkpoints in the third cell:
   ```python
   from IPython.display import FileLink
   FileLink('/kaggle/working/slidingtile_checkpoints.zip')
   ```
   *(Or click the download icon next to `slidingtile_checkpoints.zip` in the right-hand **Output** panel!)*

---

## 2. Training Specifications & What Makes It Faithful to the Paper

| Aspect | Original Paper (`train.py` / `astarlstar.py`) | Our Kaggle Implementation (`train_sliding_tile_kaggle.py`) |
| :--- | :--- | :--- |
| **Training Steps** | 20,000 steps (`range(20000)`) | Exact 20,000 steps trajectory-by-trajectory matching paper |
| **Loss Function** | $\mathcal{L}^*$ Pairwise Sibling Ranking:<br>$\text{softplus}\big((g_{\text{on}} + h_{\text{on}}) - (g_{\text{off}} + h_{\text{off}})\big)$ | Exact $\mathcal{L}^*$ pairwise sibling ranking + path monotonicity regularizer in numerically stable FP32 |
| **Precision** | FP32 | Exact FP32 with softplus clamping $[-25, 25]$ (guaranteed 0 NaNs) |
| **Optimizer** | `Adam()` re-instantiated on every step (**Bug: zero-momentum SGD**) | **Persistent PyTorch `AdamW`** with momentum, weight decay ($10^{-5}$), and `CosineAnnealingLR` |
| **Architecture** | 7-Conv DenseNet + 4 Attention Blocks + GAP + Dense 256 + Dense 1 | Exact PyTorch `ChrestienHeuristicNet(dim=5, in_channels=50)` |
| **Ensemble Training**| Single model only | **5-Model Deep Ensemble (`m0`..`m4`)**: All models trained from scratch with independent seeds, bootstrap diversity, and distinct margin scales |

---

## 3. Estimated Runtimes on Kaggle T4

- **Speed per model on Tesla T4:** $\approx 25\text{--}30$ trajectory steps/sec.
- **Training per model (20,000 steps):** $\approx 11\text{--}13$ minutes per model.
- **Total time for all 5 models:** $\approx 55\text{--}65$ minutes (well within Kaggle's 12-hour session limit).

---

## 4. Training Specific Models Individually (Optional)

If you prefer to train models one by one or in separate sessions:

- **Train Single Scratch Model #0 only:**
  ```bash
  !python -u tools/train_sliding_tile_kaggle.py --model_id 0 --epochs 25
  ```
- **Train Ensemble Models 1 and 2 only:**
  ```bash
  !python -u tools/train_sliding_tile_kaggle.py --model_id 1,2 --epochs 25
  ```

---

## 5. Integrating Downloaded Weights Back into Your Local Codebase

Once you download `slidingtile_checkpoints.zip` from Kaggle:
1. Extract the `.pt` files directly into your local `checkpoints/slidingtile/` folder:
   - `stile_ens_m0.pt`
   - `stile_ens_m1.pt`
   - `stile_ens_m2.pt`
   - `stile_ens_m3.pt`
   - `stile_ens_m4.pt`
2. You can then immediately benchmark the full 5-model ensemble on $5\times 5$, $6\times 6$, and $7\times 7$ locally:
   ```bash
   python tools/run_sliding_tile_overnight.py
   ```
