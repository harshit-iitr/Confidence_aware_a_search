import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.torch_model import ChrestienHeuristicNet
from src.sokoban_env import SokobanEnv
from src.classical_heuristics import manhattan_distance_heuristic
from src.search_algorithms import extract_lstar_training_data
from src.confidence_aware_search import get_device, author_state_to_tensor


def compute_lstar_ranking_loss(
    model: torch.nn.Module,
    on_states: List[np.ndarray],
    on_g: List[float],
    off_states: List[np.ndarray],
    off_g: List[float],
    box_targets: List[Tuple[int, int]],
    goal_state: np.ndarray,
    device: torch.device,
    dim: int = 10
) -> torch.Tensor:
    """
    Computes vectorized L* pairwise ranking loss matching Chrestien et al. (NeurIPS 2023):
    L*(h) = mean_{s+ in S+, s- in S-}( softplus( (g(s+) + h(s+)) - (g(s-) + h(s-)) ) )
    """
    on_tensors = [author_state_to_tensor(s, box_targets, dim) for s in on_states]
    on_tensor = torch.from_numpy(np.stack(on_tensors)).to(device)
    
    off_tensors = [author_state_to_tensor(s, box_targets, dim) for s in off_states]
    off_tensor = torch.from_numpy(np.stack(off_tensors)).to(device)
    
    goal_np = author_state_to_tensor(goal_state, box_targets, dim)
    goal_single = torch.from_numpy(goal_np).unsqueeze(0).to(device)
    goal_on = goal_single.expand(len(on_states), -1, -1, -1)
    goal_off = goal_single.expand(len(off_states), -1, -1, -1)
    
    h_on = model(on_tensor, goal_on)
    h_off = model(off_tensor, goal_off)
    
    g_on_t = torch.tensor(on_g, dtype=torch.float32, device=device)
    g_off_t = torch.tensor(off_g, dtype=torch.float32, device=device)
    
    f_on = g_on_t + h_on
    f_off = g_off_t + h_off
    
    # Broadcasted difference matrix: shape (len(on), len(off))
    diff = f_on.unsqueeze(1) - f_off.unsqueeze(0)
    loss = F.softplus(diff).mean()
    return loss


def train_single_ensemble_model(
    model_id: int,
    all_states: List[np.ndarray],
    bagging_ratio: float = 0.80,
    episodes: int = 2500,
    lr: float = 1e-4,
    warm_start: str = "finalSok3_pytorch.pt",
    output_dir: str = "checkpoints/ensemble",
    base_seed: int = 42,
    log_interval: int = 50,
    dim: int = 10
):
    model_seed = base_seed + model_id * 101
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"ensemble_model_{model_id}.pt")
    
    device = get_device()
    dev_name = torch.xpu.get_device_name(0) if device.type == "xpu" else (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    
    # 80% Bagging Subsampling without replacement
    n_sample = int(len(all_states) * bagging_ratio)
    rng = np.random.default_rng(model_seed)
    sampled_indices = rng.choice(len(all_states), size=n_sample, replace=False)
    bagged_states = [all_states[i] for i in sampled_indices]
    
    print("\n" + "=" * 70, flush=True)
    print(f" Training Ensemble Model Member #{model_id}", flush=True)
    print(f" Compute Device       : {device} [{dev_name}]", flush=True)
    print(f" Random Seed          : {model_seed}", flush=True)
    print(f" Bagging Subsample    : {len(bagged_states)} / {len(all_states)} instances ({bagging_ratio*100:.1f}%)", flush=True)
    print(f" Target Solved Steps  : {episodes}", flush=True)
    print(f" Learning Rate        : {lr}", flush=True)
    print(f" Warm-Start Weights   : {warm_start}", flush=True)
    print(f" Output Target        : {save_path}", flush=True)
    print("=" * 70, flush=True)
    
    model = ChrestienHeuristicNet(dim=dim).to(device)
    
    if warm_start and warm_start.lower() != "none" and os.path.exists(warm_start):
        print(f"Loading base weights from '{warm_start}' for warm-start initialization...", flush=True)
        model.load_state_dict(torch.load(warm_start, map_location=device))
        print("Warm-start weights loaded successfully.", flush=True)
    else:
        print("Initializing weights from scratch (random initialization).", flush=True)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    
    start_time = time.time()
    successful_episodes = 0
    total_loss = 0.0
    attempts = 0
    
    model.train()
    while successful_episodes < episodes:
        attempts += 1
        state = random.choice(bagged_states)
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)
        
        # Fast search to collect on-path and off-path training pairs
        data = extract_lstar_training_data(
            state, box_targets,
            lambda s, bt: manhattan_distance_heuristic(s, bt),
            max_time=2.0, max_expansions=1500, dim=dim
        )
        
        if data is None:
            continue
            
        on_states, on_g, off_states, off_g = data
        
        # Cap off-path states to prevent memory spike
        if len(off_states) > 80:
            idx = np.random.choice(len(off_states), 80, replace=False)
            off_states = [off_states[i] for i in idx]
            off_g = [off_g[i] for i in idx]
            
        optimizer.zero_grad()
        loss = compute_lstar_ranking_loss(
            model, on_states, on_g, off_states, off_g,
            box_targets, goal_state, device, dim
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_val = loss.item()
        total_loss += loss_val
        successful_episodes += 1
        
        if successful_episodes % log_interval == 0 or successful_episodes == episodes:
            elapsed = time.time() - start_time
            avg_loss = total_loss / successful_episodes
            print(f"Model #{model_id} | Step [{successful_episodes}/{episodes}] | Loss: {loss_val:.4f} | Avg Loss: {avg_loss:.4f} | Elapsed: {elapsed/60:.2f}m", flush=True)

    torch.save(model.state_dict(), save_path)
    print(f"Model #{model_id} successfully saved to '{save_path}'!\n", flush=True)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Train Deep Ensemble for Sokoban Heuristic Search")
    parser.add_argument("--train_file", type=str, default="data/sokoban_3box_train.txt", help="Path to training dataset")
    parser.add_argument("--num_models", type=int, default=5, help="Total number of ensemble models")
    parser.add_argument("--model_id", type=int, default=None, help="If set, only train this specific model (0 to num_models-1)")
    parser.add_argument("--bagging_ratio", type=float, default=0.80, help="Fraction of training instances per model (default: 0.80)")
    parser.add_argument("--episodes", type=int, default=2500, help="Number of solved training episodes per model")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--warm_start", type=str, default="finalSok3_pytorch.pt", help="Path to base checkpoint or 'none'")
    parser.add_argument("--output_dir", type=str, default="checkpoints/ensemble", help="Output directory for checkpoints")
    parser.add_argument("--base_seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--log_interval", type=int, default=50, help="Print progress every N steps")
    args = parser.parse_args()

    if not os.path.exists(args.train_file):
        alt_path = os.path.join("Optimize-Planning-Heuristics-to-Rank", "sokoban", "train", "states10_3box.txt")
        if os.path.exists(alt_path):
            args.train_file = alt_path
        else:
            raise FileNotFoundError(f"Training dataset '{args.train_file}' not found!")

    print(f"Loading training dataset from '{args.train_file}'...", flush=True)
    all_states = SokobanEnv.load_dataset(args.train_file)
    print(f"Loaded {len(all_states)} total unique training mazes.", flush=True)

    if args.model_id is not None:
        # Train a single specified model
        train_single_ensemble_model(
            model_id=args.model_id,
            all_states=all_states,
            bagging_ratio=args.bagging_ratio,
            episodes=args.episodes,
            lr=args.lr,
            warm_start=args.warm_start,
            output_dir=args.output_dir,
            base_seed=args.base_seed,
            log_interval=args.log_interval
        )
    else:
        # Train all models sequentially
        for m_id in range(args.num_models):
            train_single_ensemble_model(
                model_id=m_id,
                all_states=all_states,
                bagging_ratio=args.bagging_ratio,
                episodes=args.episodes,
                lr=args.lr,
                warm_start=args.warm_start,
                output_dir=args.output_dir,
                base_seed=args.base_seed,
                log_interval=args.log_interval
            )


if __name__ == "__main__":
    main()
