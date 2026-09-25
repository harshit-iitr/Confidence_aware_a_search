import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

from torch_model import ChrestienHeuristicNet
from sokoban_env import SokobanEnv
from classical_heuristics import manhattan_distance_heuristic
from search_algorithms import extract_lstar_training_data

def get_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def compute_lstar_loss(
    model: ChrestienHeuristicNet,
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
    Computes vectorized L* pairwise ranking loss (Chrestien et al. NeurIPS 2023):
    L*(h) = mean( softplus( (g(s+) + h(s+)) - (g(s-) + h(s-)) ) )
    """
    on_tensors = [SokobanEnv.state_to_tensor(s, box_targets, dim) for s in on_states]
    on_tensor = torch.from_numpy(np.stack(on_tensors)).to(device)
    
    off_tensors = [SokobanEnv.state_to_tensor(s, box_targets, dim) for s in off_states]
    off_tensor = torch.from_numpy(np.stack(off_tensors)).to(device)
    
    goal_tensor_single = torch.from_numpy(SokobanEnv.state_to_tensor(goal_state, box_targets, dim)).unsqueeze(0).to(device)
    goal_on = goal_tensor_single.expand(len(on_states), -1, -1, -1)
    goal_off = goal_tensor_single.expand(len(off_states), -1, -1, -1)
    
    h_on = model(on_tensor, goal_on)
    h_off = model(off_tensor, goal_off)
    
    g_on_t = torch.tensor(on_g, dtype=torch.float32, device=device)
    g_off_t = torch.tensor(off_g, dtype=torch.float32, device=device)
    
    f_on = g_on_t + h_on
    f_off = g_off_t + h_off
    
    diff = f_on.unsqueeze(1) - f_off.unsqueeze(0)
    loss = F.softplus(diff).mean()
    return loss


def train(
    train_file: str,
    num_episodes: int = 50,
    lr: float = 2e-4,
    save_path: str = "baseline_lstar.pt",
    dim: int = 10,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = get_device()
    dev_name = torch.xpu.get_device_name(0) if device.type == "xpu" else "CPU"
    print(f"==================================================", flush=True)
    print(f" Training Primary Baseline (Chrestien L* Ranking)", flush=True)
    print(f" Device: {device} [{dev_name}]", flush=True)
    print(f" Target Checkpoint: {save_path}", flush=True)
    print(f" Target Solved Episodes: {num_episodes}", flush=True)
    print(f"==================================================", flush=True)
    
    all_states = SokobanEnv.load_dataset(train_file)
    print(f"Loaded {len(all_states)} unique instances.", flush=True)
    
    model = ChrestienHeuristicNet(dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    successful_episodes = 0
    total_loss = 0.0
    attempts = 0

    model.train()
    while successful_episodes < num_episodes:
        attempts += 1
        state = random.choice(all_states)
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
        
        if len(off_states) > 80:
            idx = np.random.choice(len(off_states), 80, replace=False)
            off_states = [off_states[i] for i in idx]
            off_g = [off_g[i] for i in idx]

        optimizer.zero_grad()
        loss = compute_lstar_loss(
            model, on_states, on_g, off_states, off_g,
            box_targets, goal_state, device, dim
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_val = loss.item()
        total_loss += loss_val
        successful_episodes += 1
        
        elapsed = time.time() - start_time
        print(f"Step [{successful_episodes}/{num_episodes}] (Attempts: {attempts}) | Pairs: {len(on_states)}x{len(off_states)} | Loss: {loss_val:.4f} | Avg Loss: {total_loss/successful_episodes:.4f} | Time: {elapsed:.1f}s", flush=True)

    torch.save(model.state_dict(), save_path)
    print(f"\n==================================================", flush=True)
    print(f" Training Complete!", flush=True)
    print(f" Checkpoint: {save_path}", flush=True)
    print(f" Total Time: {time.time() - start_time:.2f}s", flush=True)
    print(f"==================================================", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chrestien et al. L* Baseline on XPU")
    parser.add_argument("--episodes", type=int, default=30, help="Number of solved training episodes")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save", type=str, default="baseline_lstar.pt", help="Path to save weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    train_file = os.path.join("Optimize-Planning-Heuristics-to-Rank", "sokoban", "train", "states10_3box.txt")
    train(train_file, num_episodes=args.episodes, lr=args.lr, save_path=args.save, seed=args.seed)
