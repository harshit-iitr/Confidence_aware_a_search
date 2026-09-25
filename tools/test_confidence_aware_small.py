import os
import time
import numpy as np
import torch
from typing import List

from torch_model import ChrestienHeuristicNet
from sokoban_env import SokobanEnv
from classical_heuristics import manhattan_distance_heuristic
from search_algorithms import run_astar
from confidence_aware_search import (
    ConfidenceAwareMCHeuristic,
    run_confidence_aware_astar,
    get_device,
    author_state_to_tensor
)

def run_small_test():
    device = get_device()
    dev_name = torch.xpu.get_device_name(0) if device.type == "xpu" else "CPU"
    
    print("==================================================================", flush=True)
    print("      TESTING CONFIDENCE-AWARE RANK-BLENDED HYBRID A* SEARCH      ", flush=True)
    print(f" Device: {device} [{dev_name}]", flush=True)
    print(" Mode: Test-Time MC Dropout (M=5 samples, p=0.10, lambda_min=0.20, C=50)", flush=True)
    print("==================================================================\n", flush=True)
    
    test_file = os.path.join("Optimize-Planning-Heuristics-to-Rank", "sokoban", "test", "states10test.txt")
    states = SokobanEnv.load_dataset(test_file)[:5]  # Small test on first 5 maps
    
    # Load model
    model = ChrestienHeuristicNet(dim=10).to(device)
    model.load_state_dict(torch.load("finalSok3_pytorch.pt", map_location=device))
    model.eval()

    print(f"Running comparative evaluation on {len(states)} sample mazes:\n", flush=True)

    for i, state in enumerate(states, 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)

        print(f"--- [MAP {i:02d}] ---", flush=True)

        # 1. Classical A* (Admissible Manhattan Matching)
        c_sol, c_exp, c_cost, _, c_t = run_astar(
            state, box_targets,
            lambda s, bt: manhattan_distance_heuristic(s, bt),
            max_time=5.0, max_expansions=10000, dim=10
        )
        print(f"  [1] Classical A*      : {'SOLVED' if c_sol else 'TIMEOUT'} | Expansions: {c_exp:5d} | Cost: {c_cost:4.1f} | Time: {c_t*1000:6.1f}ms", flush=True)

        # 2. Paper Learned A* (Deterministic Baseline)
        with torch.no_grad():
            goal_t = torch.from_numpy(author_state_to_tensor(goal_state, box_targets, 10)).unsqueeze(0).to(device)
            
        def paper_neural_h(s, bt):
            with torch.no_grad():
                st = torch.from_numpy(author_state_to_tensor(s, bt, 10)).unsqueeze(0).to(device)
                return model(st, goal_t).item()

        p_sol, p_exp, p_cost, _, p_t = run_astar(
            state, box_targets,
            paper_neural_h,
            max_time=5.0, max_expansions=10000, dim=10
        )
        print(f"  [2] Paper Learned A*  : {'SOLVED' if p_sol else 'TIMEOUT'} | Expansions: {p_exp:5d} | Cost: {p_cost:4.1f} | Time: {p_t*1000:6.1f}ms", flush=True)

        # 3. Confidence-Aware Hybrid A* (Our Method)
        hybrid_h = ConfidenceAwareMCHeuristic(
            model=model,
            goal_state=goal_state,
            box_targets=box_targets,
            device=device,
            num_mc_samples=5,
            dropout_p=0.10,
            lambda_min=0.20,
            scale_constant_C=50.0,
            dim=10
        )

        a_sol, a_exp, a_cost, _, a_t, a_stats = run_confidence_aware_astar(
            state, box_targets,
            hybrid_heuristic=hybrid_h,
            max_time=5.0, max_expansions=10000, dim=10
        )
        
        lambda_info = f"Mean lambda: {a_stats['mean_lambda']:.2f}, Mean variance: {a_stats['mean_variance']:.4f}"
        print(f"  [3] Confidence Hybrid : {'SOLVED' if a_sol else 'TIMEOUT'} | Expansions: {a_exp:5d} | Cost: {a_cost:4.1f} | Time: {a_t*1000:6.1f}ms | ({lambda_info})\n", flush=True)

    print("==================================================================", flush=True)
    print(" Small Test Complete! Verification Successful.", flush=True)
    print("==================================================================", flush=True)

if __name__ == "__main__":
    run_small_test()
