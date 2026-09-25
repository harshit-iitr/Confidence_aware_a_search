import os
import time
import argparse
import numpy as np
import torch
from typing import List, Dict

from torch_model import ChrestienHeuristicNet
from sokoban_env import SokobanEnv
from classical_heuristics import manhattan_distance_heuristic
from search_algorithms import run_astar

def get_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def make_neural_heuristic(model: ChrestienHeuristicNet, goal_state: np.ndarray, device: torch.device, dim: int = 10):
    model.eval()
    with torch.no_grad():
        goal_tensor = torch.from_numpy(SokobanEnv.state_to_tensor(goal_state, SokobanEnv.get_box_targets(goal_state), dim)).unsqueeze(0).to(device)
    
    def heuristic(state: np.ndarray, box_targets: List) -> float:
        with torch.no_grad():
            s_tensor = torch.from_numpy(SokobanEnv.state_to_tensor(state, box_targets, dim)).unsqueeze(0).to(device)
            val = model(s_tensor, goal_tensor).item()
            return float(val)
            
    return heuristic

def evaluate(
    test_file: str,
    checkpoint_path: str = "baseline_lstar.pt",
    num_instances: int = 20,
    max_time_per_instance: float = 4.0,
    dim: int = 10
):
    device = get_device()
    dev_name = torch.xpu.get_device_name(0) if device.type == "xpu" else "CPU"
    print(f"==================================================", flush=True)
    print(f" Evaluating A* Search on Test Dataset", flush=True)
    print(f" Device: {device} [{dev_name}]", flush=True)
    print(f" Checkpoint: {checkpoint_path}", flush=True)
    print(f" Test instances: {num_instances}", flush=True)
    print(f" Max Time Per Instance: {max_time_per_instance}s", flush=True)
    print(f"==================================================", flush=True)
    
    test_states = SokobanEnv.load_dataset(test_file)
    test_subset = test_states[:num_instances]
    
    has_model = os.path.exists(checkpoint_path)
    if has_model:
        model = ChrestienHeuristicNet(dim=dim).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print("Loaded neural heuristic model weights successfully.", flush=True)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running only classical baseline.", flush=True)

    classical_results = {"solved": 0, "expansions": [], "path_costs": [], "times": []}
    neural_results = {"solved": 0, "expansions": [], "path_costs": [], "times": []}

    for i, state in enumerate(test_subset, 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)

        # 1. Classical A* (Manhattan)
        c_solved, c_exp, c_cost, _, c_time = run_astar(
            state, box_targets,
            lambda s, bt: manhattan_distance_heuristic(s, bt),
            max_time=max_time_per_instance, dim=dim
        )
        if c_solved:
            classical_results["solved"] += 1
            classical_results["expansions"].append(c_exp)
            classical_results["path_costs"].append(c_cost)
            classical_results["times"].append(c_time)

        # 2. Neural L* A*
        if has_model:
            neural_h = make_neural_heuristic(model, goal_state, device, dim)
            n_solved, n_exp, n_cost, _, n_time = run_astar(
                state, box_targets,
                neural_h,
                max_time=max_time_per_instance, dim=dim
            )
            if n_solved:
                neural_results["solved"] += 1
                neural_results["expansions"].append(n_exp)
                neural_results["path_costs"].append(n_cost)
                neural_results["times"].append(n_time)

        print(f"Map {i:02d}/{num_instances} | Classical: {'SOLVED ('+str(c_exp)+' exp)' if c_solved else 'TIMEOUT'} | Neural: {'SOLVED ('+str(n_exp)+' exp)' if has_model and n_solved else 'TIMEOUT'}", flush=True)

    # Report Summary Table
    print("\n==================================================", flush=True)
    print("                BENCHMARK RESULTS                 ", flush=True)
    print("==================================================", flush=True)
    
    print("\n[1] Classical A* (Admissible Manhattan Matching):", flush=True)
    print(f"  • Solved: {classical_results['solved']} / {num_instances} ({classical_results['solved']/num_instances*100:.1f}%)", flush=True)
    if classical_results["expansions"]:
        print(f"  • Node Expansions (Mean ± Std): {np.mean(classical_results['expansions']):.1f} ± {np.std(classical_results['expansions']):.1f}", flush=True)
        print(f"  • Median Expansions: {np.median(classical_results['expansions']):.1f}", flush=True)
        print(f"  • Mean Path Cost: {np.mean(classical_results['path_costs']):.2f}", flush=True)
        print(f"  • Mean Search Time: {np.mean(classical_results['times'])*1000:.1f} ms", flush=True)

    if has_model and neural_results["solved"] > 0:
        print("\n[2] Learned A* (Chrestien et al. L* Ranking Baseline):", flush=True)
        print(f"  • Solved: {neural_results['solved']} / {num_instances} ({neural_results['solved']/num_instances*100:.1f}%)", flush=True)
        print(f"  • Node Expansions (Mean ± Std): {np.mean(neural_results['expansions']):.1f} ± {np.std(neural_results['expansions']):.1f}", flush=True)
        print(f"  • Median Expansions: {np.median(neural_results['expansions']):.1f}", flush=True)
        print(f"  • Mean Path Cost: {np.mean(neural_results['path_costs']):.2f}", flush=True)
        print(f"  • Mean Search Time: {np.mean(neural_results['times'])*1000:.1f} ms", flush=True)
        
        if classical_results["expansions"]:
            exp_ratio = np.mean(neural_results['expansions']) / np.mean(classical_results['expansions'])
            print(f"\n>> Expansion Ratio (Neural / Classical): {exp_ratio:.2f}x", flush=True)
    print("==================================================\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Classical vs. Learned A*")
    parser.add_argument("--test_file", type=str, default=os.path.join("Optimize-Planning-Heuristics-to-Rank", "sokoban", "test", "states10test.txt"))
    parser.add_argument("--checkpoint", type=str, default="baseline_lstar.pt")
    parser.add_argument("--n_samples", type=int, default=15)
    args = parser.parse_args()
    
    evaluate(args.test_file, checkpoint_path=args.checkpoint, num_instances=args.n_samples)
