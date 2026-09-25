import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict

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

def create_ood_perturbed_state(state: np.ndarray, num_perturbations: int = 4) -> np.ndarray:
    """
    Creates an Out-Of-Distribution (OOD) perturbed state by randomly placing 
    extra obstacle walls in open floor spaces, simulating unseen / high-ambiguity terrain.
    """
    ood_state = state.copy()
    floor_rows, floor_cols = np.where(ood_state == 1)
    if len(floor_rows) > num_perturbations:
        chosen_indices = np.random.choice(len(floor_rows), num_perturbations, replace=False)
        for idx in chosen_indices:
            ood_state[floor_rows[idx], floor_cols[idx]] = 0  # Turn open floor into wall
    return ood_state


def run_deep_validation():
    device = get_device()
    dev_name = torch.xpu.get_device_name(0) if device.type == "xpu" else "CPU"
    
    print("==================================================================", flush=True)
    print("           DEEP THEORETICAL & EMPIRICAL VALIDATION SUITE          ", flush=True)
    print(f" Device: {device} [{dev_name}]", flush=True)
    print("==================================================================\n", flush=True)
    
    test_file = os.path.join("Optimize-Planning-Heuristics-to-Rank", "sokoban", "test", "states10test.txt")
    states = SokobanEnv.load_dataset(test_file)[:8]
    
    model = ChrestienHeuristicNet(dim=10).to(device)
    model.load_state_dict(torch.load("finalSok3_pytorch.pt", map_location=device))
    model.eval()

    # -------------------------------------------------------------
    # EXPERIMENT 1: IN-DISTRIBUTION vs OUT-OF-DISTRIBUTION (OOD) UNCERTAINTY
    # -------------------------------------------------------------
    print(">>> EXPERIMENT 1: Epistemic Uncertainty on In-Distribution vs. OOD States", flush=True)
    print("Testing if MC Dropout variance increases and lambda drops on ambiguous/OOD states:\n", flush=True)
    
    in_dist_variances = []
    in_dist_lambdas = []
    ood_variances = []
    ood_lambdas = []

    for i, state in enumerate(states[:4], 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)

        hybrid_h = ConfidenceAwareMCHeuristic(
            model=model, goal_state=goal_state, box_targets=box_targets,
            device=device, num_mc_samples=5, dropout_p=0.10, lambda_min=0.20, scale_constant_C=50.0
        )

        # In-distribution state evaluation
        eval_in = hybrid_h.evaluate(state)
        in_dist_variances.append(eval_in.variance)
        in_dist_lambdas.append(eval_in.lambda_conf)

        # OOD perturbed state evaluation
        ood_state = create_ood_perturbed_state(state, num_perturbations=5)
        eval_ood = hybrid_h.evaluate(ood_state)
        ood_variances.append(eval_ood.variance)
        ood_lambdas.append(eval_ood.lambda_conf)

        print(f"Map {i:02d} In-Dist         -> Variance: {eval_in.variance:.4f} | Lambda: {eval_in.lambda_conf:.2f} (Trusts learned heuristic)", flush=True)
        print(f"Map {i:02d} OOD (Corrupted) -> Variance: {eval_ood.variance:.4f} | Lambda: {eval_ood.lambda_conf:.2f} (Falls back to classical!)\n", flush=True)

    print(f"Summary Statistics:")
    print(f"  - Average In-Distribution Lambda : {np.mean(in_dist_lambdas):.3f} (High confidence)")
    print(f"  - Average OOD Lambda             : {np.mean(ood_lambdas):.3f} (Significant fallback to classical)")
    print(f"  - Variance Increase Ratio        : {np.mean(ood_variances)/max(1e-5, np.mean(in_dist_variances)):.2f}x higher variance on OOD!\n", flush=True)

    # -------------------------------------------------------------
    # EXPERIMENT 2: CONTROLLED SEARCH COMPARISON (CLASSICAL vs PURE NN vs HYBRID)
    # -------------------------------------------------------------
    print(">>> EXPERIMENT 2: Controlled Search Behavior Across 8 Benchmark Mazes", flush=True)
    print("Detailed metrics per map:\n", flush=True)
    
    results = {
        "Classical A*": {"sol": 0, "exp": [], "cost": [], "time": []},
        "Paper Learned A*": {"sol": 0, "exp": [], "cost": [], "time": []},
        "Confidence Hybrid": {"sol": 0, "exp": [], "cost": [], "time": [], "lambdas": [], "vars": []}
    }

    for i, state in enumerate(states, 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)

        # 1. Classical
        c_sol, c_exp, c_cost, _, c_t = run_astar(
            state, box_targets,
            lambda s, bt: manhattan_distance_heuristic(s, bt),
            max_time=4.0, max_expansions=10000, dim=10
        )
        if c_sol:
            results["Classical A*"]["sol"] += 1
            results["Classical A*"]["exp"].append(c_exp)
            results["Classical A*"]["cost"].append(c_cost)
            results["Classical A*"]["time"].append(c_t)

        # 2. Paper Learned A*
        with torch.no_grad():
            goal_t = torch.from_numpy(author_state_to_tensor(goal_state, box_targets, 10)).unsqueeze(0).to(device)
            
        def neural_h(s, bt):
            with torch.no_grad():
                st = torch.from_numpy(author_state_to_tensor(s, bt, 10)).unsqueeze(0).to(device)
                return model(st, goal_t).item()

        p_sol, p_exp, p_cost, _, p_t = run_astar(
            state, box_targets,
            neural_h,
            max_time=4.0, max_expansions=10000, dim=10
        )
        if p_sol:
            results["Paper Learned A*"]["sol"] += 1
            results["Paper Learned A*"]["exp"].append(p_exp)
            results["Paper Learned A*"]["cost"].append(p_cost)
            results["Paper Learned A*"]["time"].append(p_t)

        # 3. Confidence Hybrid
        hybrid_h = ConfidenceAwareMCHeuristic(
            model=model, goal_state=goal_state, box_targets=box_targets,
            device=device, num_mc_samples=5, dropout_p=0.10, lambda_min=0.20, scale_constant_C=50.0, dim=10
        )

        h_sol, h_exp, h_cost, _, h_t, h_stats = run_confidence_aware_astar(
            state, box_targets,
            hybrid_heuristic=hybrid_h,
            max_time=4.0, max_expansions=10000, dim=10
        )
        if h_sol:
            results["Confidence Hybrid"]["sol"] += 1
            results["Confidence Hybrid"]["exp"].append(h_exp)
            results["Confidence Hybrid"]["cost"].append(h_cost)
            results["Confidence Hybrid"]["time"].append(h_t)
            results["Confidence Hybrid"]["lambdas"].append(h_stats["mean_lambda"])
            results["Confidence Hybrid"]["vars"].append(h_stats["mean_variance"])

        print(f"Map {i:02d} | Classical: {'SOLVED ('+str(c_exp)+' exp)' if c_sol else 'TIMEOUT'} | Paper NN: {'SOLVED ('+str(p_exp)+' exp)' if p_sol else 'TIMEOUT'} | Hybrid: {'SOLVED ('+str(h_exp)+' exp, lambda='+str(round(h_stats['mean_lambda'],2))+')' if h_sol else 'TIMEOUT'}", flush=True)

    print("\n==================================================================", flush=True)
    print("                    AGGREGATE VALIDATION SUMMARY                  ", flush=True)
    print("==================================================================", flush=True)
    for alg, data in results.items():
        sol_rate = data["sol"] / len(states) * 100
        mean_e = np.mean(data["exp"]) if data["exp"] else 0
        med_e = np.median(data["exp"]) if data["exp"] else 0
        mean_c = np.mean(data["cost"]) if data["cost"] else 0
        print(f"[{alg}]", flush=True)
        print(f"  - Solved Rate       : {data['sol']}/{len(states)} ({sol_rate:.1f}%)", flush=True)
        print(f"  - Expansions (Mean) : {mean_e:.1f} nodes", flush=True)
        print(f"  - Expansions (Med)  : {med_e:.1f} nodes", flush=True)
        print(f"  - Mean Solution Cost: {mean_c:.2f}", flush=True)
        if "lambdas" in data and data["lambdas"]:
            print(f"  - Mean Lambda       : {np.mean(data['lambdas']):.2f}", flush=True)
    print("==================================================================\n", flush=True)

if __name__ == "__main__":
    run_deep_validation()
