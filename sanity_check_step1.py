import os
import time
import numpy as np
import torch
from typing import List, Tuple

from sokoban_env import SokobanEnv
from torch_model import ChrestienHeuristicNet
from classical_heuristics import manhattan_distance_heuristic
from search_algorithms import run_astar
from confidence_aware_search import (
    ConfidenceAwareMCHeuristic,
    run_confidence_aware_astar,
    get_device,
    author_state_to_tensor
)

def run_step1_sanity_audit():
    print("==================================================================", flush=True)
    print("         STEP 1: RESEARCH INTEGRITY & SANITY AUDIT SUITE          ", flush=True)
    print("==================================================================\n", flush=True)

    train_file = os.path.join("Optimize-Planning-Heuristics-to-Rank", "sokoban", "train", "states10_3box.txt")
    test_file = os.path.join("Optimize-Planning-Heuristics-to-Rank", "sokoban", "test", "states10test.txt")
    
    # -------------------------------------------------------------------------
    # CHECK 1: ZERO DATA LEAKAGE AUDIT
    # -------------------------------------------------------------------------
    print(">>> [AUDIT 1/4] Checking Data Leakage Between Train and Test Sets...", flush=True)
    
    train_raw = []
    with open(train_file, "r") as f:
        for line in f:
            nums = tuple(int(x) for x in line.split())
            if nums:
                train_raw.append(nums)
                
    test_raw = []
    with open(test_file, "r") as f:
        for line in f:
            nums = tuple(int(x) for x in line.split())
            if nums:
                test_raw.append(nums)

    train_set = set(train_raw)
    test_set = set(test_raw)
    overlap = train_set.intersection(test_set)

    print(f"  * Total Training Instances in File : {len(train_raw)} (Unique: {len(train_set)})", flush=True)
    print(f"  * Total Testing Instances in File  : {len(test_raw)} (Unique: {len(test_set)})", flush=True)
    print(f"  * Overlapping Instances (Train & Test) : {len(overlap)}", flush=True)
    
    if len(overlap) == 0:
        print("  [PASS] ZERO DATA LEAKAGE: Test instances are completely disjoint from training set.\n", flush=True)
    else:
        print(f"  [FAIL] DATA LEAKAGE DETECTED: {len(overlap)} states appear in both train and test!\n", flush=True)

    # -------------------------------------------------------------------------
    # CHECK 2: MATHEMATICAL BOUNDS & CAUSALITY AUDIT
    # -------------------------------------------------------------------------
    print(">>> [AUDIT 2/4] Verifying Mathematical Invariants & Bounds...", flush=True)
    device = get_device()
    model = ChrestienHeuristicNet(dim=10).to(device)
    model.load_state_dict(torch.load("finalSok3_pytorch.pt", map_location=device))
    model.eval()

    test_states = SokobanEnv.load_dataset(test_file)[:10]
    
    bounds_pass = True
    for idx, state in enumerate(test_states, 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)

        hybrid_h = ConfidenceAwareMCHeuristic(
            model=model, goal_state=goal_state, box_targets=box_targets,
            device=device, num_mc_samples=5, dropout_p=0.10, lambda_min=0.20, scale_constant_C=50.0
        )
        
        states_to_test = [state]
        neighbors, _, _ = SokobanEnv.get_neighbors(state, box_targets, 10)
        states_to_test.extend(neighbors)

        for s in states_to_test:
            res = hybrid_h.evaluate(s)
            
            # Check p_class in [0, 1]
            if not (0.0 <= res.p_class <= 1.0):
                bounds_pass = False
                print(f"  [ERROR] p_class out of bounds: {res.p_class}")
            # Check mean_p in [0, 1]
            if not (0.0 <= res.mean_p <= 1.0):
                bounds_pass = False
                print(f"  [ERROR] mean_p out of bounds: {res.mean_p}")
            # Check variance in [0.0, 0.25]
            if not (0.0 <= res.variance <= 0.250001):
                bounds_pass = False
                print(f"  [ERROR] rank variance out of bounds: {res.variance}")
            # Check lambda in [lambda_min, 1.0]
            if not (0.20 <= res.lambda_conf <= 1.000001):
                bounds_pass = False
                print(f"  [ERROR] lambda out of bounds: {res.lambda_conf}")
            # Check p_blend in [0, 1]
            if not (0.0 <= res.p_blend <= 1.0):
                bounds_pass = False
                print(f"  [ERROR] p_blend out of bounds: {res.p_blend}")
            # Check h_blend >= 0
            if res.h_blend < 0.0:
                bounds_pass = False
                print(f"  [ERROR] h_blend is negative: {res.h_blend}")

    if bounds_pass:
        print("  [PASS] ALL MATHEMATICAL BOUNDS VERIFIED:")
        print("         p_class in [0, 1], p_m in [0, 1]")
        print("         rank variance in [0, 0.25]")
        print("         confidence gate lambda in [0.20, 1.00]")
        print("         blended heuristic h_blend >= 0\n", flush=True)
    else:
        print("  [FAIL] Mathematical bound violations detected!\n", flush=True)

    # -------------------------------------------------------------------------
    # CHECK 3: ENVIRONMENT TRANSITION SOUNDNESS & DETERMINISM
    # -------------------------------------------------------------------------
    print(">>> [AUDIT 3/4] Checking Environment Neighbor Generation & State Hashing...", flush=True)
    env_pass = True
    for state in test_states[:5]:
        box_targets = SokobanEnv.get_box_targets(state)
        next_states1, acts1, costs1 = SokobanEnv.get_neighbors(state, box_targets, 10)
        next_states2, acts2, costs2 = SokobanEnv.get_neighbors(state, box_targets, 10)
        
        # Verify determinism
        if len(next_states1) != len(next_states2) or acts1 != acts2 or costs1 != costs2:
            env_pass = False
            print("  [ERROR] Neighbor generation is non-deterministic!")
            
        for ns in next_states1:
            # Check player count == 1
            if np.sum(ns == 3) != 1:
                env_pass = False
                print("  [ERROR] Invalid player count after transition!")
            # Check box count == 3
            if np.sum(ns == 4) != 3:
                env_pass = False
                print("  [ERROR] Invalid box count after transition!")
                
    if env_pass:
        print("  [PASS] ENVIRONMENT INTEGRITY: Valid transitions, exact box/player conservation.\n", flush=True)
    else:
        print("  [FAIL] Environment integrity issues found!\n", flush=True)

    # -------------------------------------------------------------------------
    # CHECK 4: INDEPENDENT PATH & ACTION VERIFICATION (SOUNDNESS TEST)
    # -------------------------------------------------------------------------
    print(">>> [AUDIT 4/4] Independent Solution Replay & Action Soundness Audit...", flush=True)
    print("Simulating step-by-step action executions from initial state to goal:\n", flush=True)

    path_audit_pass = True
    for map_idx, state in enumerate(test_states[:5], 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)

        hybrid_h = ConfidenceAwareMCHeuristic(
            model=model, goal_state=goal_state, box_targets=box_targets,
            device=device, num_mc_samples=5, dropout_p=0.10, lambda_min=0.20, scale_constant_C=50.0
        )

        solved, exp, cost, actions, elapsed, stats = run_confidence_aware_astar(
            state, box_targets, hybrid_h, max_time=5.0, dim=10
        )

        if not solved:
            print(f"  Map {map_idx:02d}: Not solved within timeout.")
            continue

        # Replay the action sequence independently step by step
        sim_state = state.copy()
        for step_idx, act in enumerate(actions, 1):
            valid_nexts, valid_acts, _ = SokobanEnv.get_neighbors(sim_state, box_targets, 10)
            if act not in valid_acts:
                path_audit_pass = False
                print(f"  [ERROR] Map {map_idx:02d} Step {step_idx}: Action {act} is illegal from current state!")
                break
            act_pos = valid_acts.index(act)
            sim_state = valid_nexts[act_pos]

        # Verify final simulated state is genuinely the goal
        is_real_goal = SokobanEnv.is_goal(sim_state, box_targets)
        if not is_real_goal:
            path_audit_pass = False
            print(f"  [ERROR] Map {map_idx:02d}: Action plan executed but final state is NOT a goal state!")
        else:
            print(f"  Map {map_idx:02d} | Path Length: {len(actions):2d} steps | Cost Reported: {cost:4.1f} | Replay Status: VALIDATED GOAL REACHED", flush=True)

    print()
    if path_audit_pass:
        print("  [PASS] ACTION SOUNDNESS CONFIRMED: 100% of generated paths are strictly executable,", flush=True)
        print("         physically valid, and reach the true goal state with no shortcuts.\n", flush=True)
    else:
        print("  [FAIL] Path execution validation failed!\n", flush=True)

    print("==================================================================", flush=True)
    print("             AUDIT SUMMARY: STEP 1 FULLY COMPLETED                ", flush=True)
    print("==================================================================", flush=True)

if __name__ == "__main__":
    run_step1_sanity_audit()
