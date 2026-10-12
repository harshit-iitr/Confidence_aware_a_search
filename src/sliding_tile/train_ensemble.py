import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.torch_model import ChrestienHeuristicNet
from src.sliding_tile.env import (
    get_goal_board,
    load_sliding_tile_boards,
    manhattan_linear_conflict,
    get_valid_neighbors,
    encode_boards_25ch,
)

_EYE_25 = np.eye(25, dtype=np.float32)


def generate_biased_walk_trajectory(goal: np.ndarray, rng: np.random.Generator, walk_len: int) -> list:
    """
    Generates a non-backtracking walk from `goal` of length `walk_len`, biased toward
    increasing Manhattan+LinearConflict distance so trajectories span MD+LC in [5, 95].
    Returns list of (board, step_dist_from_goal, is_on_path, parent_step_dist).
    """
    cur = goal.copy()
    path_from_goal = [cur.copy()]
    prev_r, prev_c = -1, -1

    for _ in range(walk_len):
        pos = np.argwhere(cur == 0)[0]
        br, bc = int(pos[0]), int(pos[1])
        cands = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = br + dr, bc + dc
            if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) != (prev_r, prev_c):
                nb = cur.copy()
                nb[br, bc] = nb[nr, nc]
                nb[nr, nc] = 0
                cands.append((manhattan_linear_conflict(nb, 5), nb, nr, nc))
        cands.sort(key=lambda x: x[0], reverse=True)
        idx = 0 if (rng.random() < 0.72 or len(cands) == 1) else int(rng.integers(1, len(cands)))
        prev_r, prev_c = br, bc
        cur = cands[idx][1]
        path_from_goal.append(cur.copy())

    # Subsample 8 states along the walk + their off-path sibling children
    L = len(path_from_goal) - 1
    chosen_d = sorted(set([0, 1, L] + rng.choice(range(1, max(2, L)), size=min(6, max(1, L - 1)), replace=False).tolist()))
    records = []
    path_bytes = {b.tobytes() for b in path_from_goal}
    for d in chosen_d:
        b = path_from_goal[d]
        md = manhattan_linear_conflict(b, 5)
        records.append((b, float(d), float(md), 1))
        if d > 0:
            for nb, _ in get_valid_neighbors(b, 5):
                if nb.tobytes() not in path_bytes:
                    records.append((nb, float(d + 1), float(manhattan_linear_conflict(nb, 5)), 0))
                    break
    return records


def build_training_groups(
    cache_path: str,
    device: torch.device,
    num_walks: int = 800,
    num_stile_boards: int = 1200,
) -> dict:
    """
    Constructs or loads grouped ranking blocks where each group contains ~16 states
    (on-path trajectory states + off-path sibling states + real train/stile.txt local trees)
    annotated with structural cost-to-go rankings and teacher reference scores.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached Sliding-Tile training groups from {cache_path}...")
        return torch.load(cache_path, weights_only=False)

    print("Building Sliding-Tile L* ranking dataset from train/stile.txt + biased goal walks...")
    t0 = time.time()
    goal = get_goal_board(5)
    rng = np.random.default_rng(2026)

    groups_boards = []
    groups_md = []
    groups_on_flag = []

    # Part 1: Biased walks from goal (exact path + sibling off-path states)
    for _ in range(num_walks):
        w_len = int(rng.integers(12, 135))
        recs = generate_biased_walk_trajectory(goal, rng, w_len)
        # Pad or trim to 12 states per group for uniform fast batching
        while len(recs) < 12:
            recs.append(recs[-1])
        recs = recs[:12]
        groups_boards.append(np.stack([r[0] for r in recs], axis=0))
        groups_md.append(np.array([r[2] for r in recs], dtype=np.float32))
        groups_on_flag.append(np.array([r[3] for r in recs], dtype=np.float32))

    # Part 2: Real boards from train/stile.txt + local 1-step & 2-step expansion trees
    stile_train_path = "Optimize-Planning-Heuristics-to-Rank/slidingtile/train/stile.txt"
    train_boards = load_sliding_tile_boards(stile_train_path, 5)[:num_stile_boards]
    for b0 in train_boards:
        local_list = [(b0, manhattan_linear_conflict(b0, 5), 1.0)]
        nbrs1 = get_valid_neighbors(b0, 5)
        nbrs1_scored = [(nb, manhattan_linear_conflict(nb, 5)) for nb, _ in nbrs1]
        nbrs1_scored.sort(key=lambda x: x[1])
        # Best child is on-path candidate (o=1), others are off-path siblings (o=0)
        for i, (nb, md_v) in enumerate(nbrs1_scored):
            local_list.append((nb, md_v, 1.0 if i == 0 else 0.0))
            for nb2, _ in get_valid_neighbors(nb, 5):
                if nb2.tobytes() != b0.tobytes():
                    local_list.append((nb2, manhattan_linear_conflict(nb2, 5), 0.0))
        # Also include goal and a near-goal state in every group to anchor zero-distance ranking
        local_list.append((goal, 0.0, 1.0))
        while len(local_list) < 12:
            local_list.append(local_list[-1])
        local_list = local_list[:12]
        groups_boards.append(np.stack([x[0] for x in local_list], axis=0))
        groups_md.append(np.array([x[1] for x in local_list], dtype=np.float32))
        groups_on_flag.append(np.array([x[2] for x in local_list], dtype=np.float32))

    boards_arr = np.asarray(groups_boards, dtype=np.uint8)      # (G, 12, 5, 5)
    md_arr = np.asarray(groups_md, dtype=np.float32)            # (G, 12)
    on_arr = np.asarray(groups_on_flag, dtype=np.float32)       # (G, 12)

    # Compute teacher reference ranking values using converted stile checkpoint
    teacher = ChrestienHeuristicNet(dim=5, in_channels=50).to(device)
    teacher.load_state_dict(torch.load("checkpoints/slidingtile/stile_paper_pytorch.pt", map_location=device))
    teacher.eval()

    G, K, _, _ = boards_arr.shape
    flat_boards = boards_arr.reshape(G * K, 5, 5)
    teacher_preds = np.zeros(G * K, dtype=np.float32)
    goal_t = encode_boards_25ch([goal], device)

    batch_sz = 256
    with torch.no_grad():
        g_zero = float(teacher(goal_t, goal_t)[0].item())
        for start_i in range(0, G * K, batch_sz):
            end_i = min(G * K, start_i + batch_sz)
            s_t = encode_boards_25ch(flat_boards[start_i:end_i], device)
            gb_t = goal_t.expand(end_i - start_i, -1, -1, -1)
            out = teacher(s_t, gb_t).cpu().numpy() - g_zero
            teacher_preds[start_i:end_i] = np.maximum(0.0, out)

    teacher_arr = teacher_preds.reshape(G, K)
    data = {
        "boards": boards_arr,
        "md": md_arr,
        "on_flag": on_arr,
        "teacher": teacher_arr,
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(data, cache_path)
    print(f"Built and cached {G} ranking groups ({G*K} states) in {time.time() - t0:.1f}s.")
    return data


def compute_batched_lstar_loss(
    preds: torch.Tensor,
    target_rank_score: torch.Tensor,
    on_flag: torch.Tensor,
    margin_scale: float = 1.0,
) -> torch.Tensor:
    """
    Computes batched pairwise L* ranking loss + scale-calibrated margin loss across (B, K) groups:
      1. Pairwise L* ranking within each group of K=12 states:
         For all pairs (i, j) where state i is strictly closer than state j (s_j - s_i > 1.5),
         enforce h(s_i) < h(s_j) via softplus(h(s_i) - h(s_j) + margin_scale * 0.45 * (s_j - s_i))
      2. Anchor loss keeping predictions aligned with the L* ranking scale.
    """
    s_i = target_rank_score.unsqueeze(2)  # (B, K, 1)
    s_j = target_rank_score.unsqueeze(1)  # (B, 1, K)
    p_i = preds.unsqueeze(2)              # (B, K, 1)
    p_j = preds.unsqueeze(1)              # (B, 1, K)

    # diff[b, i, j] = s_j - s_i > 1.5 means state i is CLOSER to goal than state j
    diff = s_j - s_i
    mask = diff > 1.5
    margin = margin_scale * 0.45 * diff
    rank_loss = F.softplus(p_i - p_j + margin)[mask].mean()

    huber_anchor = F.smooth_l1_loss(preds, margin_scale * target_rank_score, beta=6.0)
    return 0.70 * rank_loss + 0.30 * huber_anchor


def train_sliding_tile_ensemble(
    num_models: int = 5,
    epochs: int = 12,
    batch_groups: int = 16,
    lr: float = 1.5e-4,
    out_dir: str = "checkpoints/slidingtile",
):
    device = torch.device(
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"=== Training Sliding-Tile {num_models}-Model Deep Ensemble on {device} ===")
    os.makedirs(out_dir, exist_ok=True)

    data = build_training_groups(
        os.path.join(out_dir, "lstar_groups_5x5.pt"),
        device=device,
        num_walks=800,
        num_stile_boards=1200,
    )

    boards_all = data["boards"]    # (G, 12, 5, 5)
    md_all = data["md"]            # (G, 12)
    on_all = data["on_flag"]       # (G, 12)
    teacher_all = data["teacher"]  # (G, 12)

    G, K, _, _ = boards_all.shape
    n_val = 200
    n_train = G - n_val

    goal = get_goal_board(5)
    goal_t = encode_boards_25ch([goal], device)

    # Distinct margin scales & MD+LC blend weights per ensemble member for genuine multi-basin diversity
    margin_scales = [0.92, 0.96, 1.00, 1.05, 1.10]
    md_mix_weights = [0.04, 0.10, 0.16, 0.22, 0.28]

    for m_idx in range(num_models):
        ckpt_path = os.path.join(out_dir, f"stile_ens_m{m_idx}.pt")
        if os.path.exists(ckpt_path):
            print(f"[Model {m_idx}] Checkpoint {ckpt_path} already exists, skipping.")
            continue

        t_start = time.time()
        seed = 3000 + m_idx * 137
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        model = ChrestienHeuristicNet(dim=5, in_channels=50).to(device)
        base_sd = torch.load("checkpoints/slidingtile/stile_paper_pytorch.pt", map_location=device)
        my_sd = model.state_dict()
        for k_name, tensor in base_sd.items():
            if any(k_name.startswith(p) for p in ("conv9", "conv10", "conv11", "fc1", "fc2")):
                # Perturb high-level attention & dense heads with model-specific seed to induce basin diversity
                std_t = float(tensor.std().item()) if tensor.numel() > 1 else 0.05
                noise_scale = 0.05 + 0.03 * m_idx
                my_sd[k_name] = tensor + torch.randn_like(tensor) * (std_t * noise_scale)
            else:
                my_sd[k_name] = tensor.clone()
        model.load_state_dict(my_sd)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        # Each model trains on an independent 80% bootstrap subset and distinct blend of L* + MD+LC
        w_md = md_mix_weights[m_idx % len(md_mix_weights)]
        m_scale = margin_scales[m_idx % len(margin_scales)]
        rank_scores_all = (1.0 - w_md) * teacher_all + w_md * (2.2 * md_all)

        bag_size = int(0.80 * n_train)
        bag_indices = rng.choice(n_train, size=bag_size, replace=False)

        best_val_loss = float("inf")
        for ep in range(1, epochs + 1):
            model.train()
            rng.shuffle(bag_indices)
            ep_loss = 0.0
            n_batches = 0

            for b_start in range(0, bag_size, batch_groups):
                b_idxs = bag_indices[b_start:b_start + batch_groups]
                B_cur = len(b_idxs)
                b_boards = boards_all[b_idxs].reshape(B_cur * K, 5, 5)
                s_t = encode_boards_25ch(b_boards, device)
                gb_t = goal_t.expand(B_cur * K, -1, -1, -1)

                preds = model(s_t, gb_t, mc_dropout_p=0.10).view(B_cur, K)
                t_scores = torch.from_numpy(rank_scores_all[b_idxs]).to(device)
                t_on = torch.from_numpy(on_all[b_idxs]).to(device)

                loss = compute_batched_lstar_loss(preds, t_scores, t_on, margin_scale=m_scale)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

                ep_loss += float(loss.item())
                n_batches += 1

            scheduler.step()
            train_loss_avg = ep_loss / max(1, n_batches)

            model.eval()
            val_loss = 0.0
            val_batches = 0
            all_v_preds, all_v_targets = [], []
            with torch.no_grad():
                for v_start in range(n_train, G, batch_groups):
                    v_end = min(G, v_start + batch_groups)
                    B_v = v_end - v_start
                    v_boards = boards_all[v_start:v_end].reshape(B_v * K, 5, 5)
                    s_t = encode_boards_25ch(v_boards, device)
                    gb_t = goal_t.expand(B_v * K, -1, -1, -1)
                    v_preds = model(s_t, gb_t, mc_dropout_p=0.0).view(B_v, K)
                    v_scores = torch.from_numpy(rank_scores_all[v_start:v_end]).to(device)
                    v_on = torch.from_numpy(on_all[v_start:v_end]).to(device)
                    v_l = compute_batched_lstar_loss(v_preds, v_scores, v_on, margin_scale=m_scale)
                    val_loss += float(v_l.item())
                    val_batches += 1
                    all_v_preds.append(v_preds.cpu().numpy().ravel())
                    all_v_targets.append(v_scores.cpu().numpy().ravel())

            val_loss /= max(1, val_batches)
            corr = float(np.corrcoef(np.concatenate(all_v_preds), np.concatenate(all_v_targets))[0, 1])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), ckpt_path)

            if ep == 1 or ep % 3 == 0 or ep == epochs:
                print(
                    f"[Model {m_idx}] Epoch {ep:2d}/{epochs} | "
                    f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val r: {corr:.4f} | Time: {time.time() - t_start:.1f}s",
                    flush=True,
                )

        print(f"Saved best Model {m_idx} checkpoint to {ckpt_path} ({time.time() - t_start:.1f}s)\n")


if __name__ == "__main__":
    train_sliding_tile_ensemble()
