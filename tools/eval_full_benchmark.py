import os
import time
import argparse
import numpy as np
import torch
import heapq
from typing import List, Dict

from torch_model import ChrestienHeuristicNet
from sokoban_env import SokobanEnv
from classical_heuristics import manhattan_distance_heuristic

def get_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class SearchNode:
    __slots__ = ('state', 'key', 'g', 'h', 'f', 'parent_key', 'action')
    def __init__(self, state: np.ndarray, g: float, h: float, f: float, parent_key=None, action=0):
        self.state = state
        self.key = SokobanEnv.state_to_key(state)
        self.g = g
        self.h = h
        self.f = f
        self.parent_key = parent_key
        self.action = action

def run_search(
    init_state: np.ndarray,
    box_targets: List,
    heuristic_fn,
    alg: str = "astar",
    max_expansions: int = 10000,
    max_time: float = 10.0,
    dim: int = 10
):
    start_time = time.time()
    init_h = heuristic_fn(init_state, box_targets)
    init_f = init_h if alg == "gbfs" else (0.0 + init_h)
    start_node = SearchNode(init_state, g=0.0, h=init_h, f=init_f)
    
    open_heap = []
    heapq.heappush(open_heap, (start_node.f, start_node.h, 0, start_node))
    
    open_dict = {start_node.key: start_node.g}
    closed_dict = {}
    
    counter = 0
    expansions = 0
    
    while open_heap:
        if (time.time() - start_time) > max_time or expansions >= max_expansions:
            return False, expansions, float("inf"), time.time() - start_time
            
        _, _, _, current = heapq.heappop(open_heap)
        if current.key in closed_dict:
            continue
        closed_dict[current.key] = current
        expansions += 1
        
        if SokobanEnv.is_goal(current.state, box_targets):
            return True, expansions, current.g, time.time() - start_time
            
        next_states, act_nos, costs = SokobanEnv.get_neighbors(current.state, box_targets, dim)
        for n_state, act, cost in zip(next_states, act_nos, costs):
            n_key = SokobanEnv.state_to_key(n_state)
            new_g = current.g + cost
            
            if n_key in closed_dict:
                continue
                
            if n_key not in open_dict or new_g < open_dict[n_key]:
                open_dict[n_key] = new_g
                h_val = heuristic_fn(n_state, box_targets)
                f_val = h_val if alg == "gbfs" else (new_g + h_val)
                child = SearchNode(n_state, g=new_g, h=h_val, f=f_val, parent_key=current.key, action=act)
                counter += 1
                heapq.heappush(open_heap, (child.f, child.h, counter, child))
                
    return False, expansions, float("inf"), time.time() - start_time

def evaluate_full(test_file: str, checkpoint_path: str = "finalSok3_pytorch.pt", n_samples: int = 30):
    device = get_device()
    dev_name = torch.xpu.get_device_name(0) if device.type == "xpu" else "CPU"
    print(f"==================================================", flush=True)
    print(f" BENCHMARK: Paper Pre-trained finalSok3 Checkpoint", flush=True)
    print(f" Device: {device} [{dev_name}]", flush=True)
    print(f" Checkpoint: {checkpoint_path}", flush=True)
    print(f" Test Instances: {n_samples}", flush=True)
    print(f"==================================================", flush=True)
    
    test_states = SokobanEnv.load_dataset(test_file)[:n_samples]
    
    model = ChrestienHeuristicNet(dim=10).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Pre-build tensor cache
    results = {
        "Classical A*": {"solved": 0, "exp": [], "time": []},
        "Paper Learned A*": {"solved": 0, "exp": [], "time": []},
        "Paper Learned GBFS": {"solved": 0, "exp": [], "time": []},
    }

    for i, state in enumerate(test_states, 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)
        
        goal_tensor = torch.from_numpy(SokobanEnv.state_to_tensor(goal_state, box_targets, 10)).unsqueeze(0).to(device)
        
        def neural_h(s, bt):
            with torch.no_grad():
                st = torch.from_numpy(SokobanEnv.state_to_tensor(s, bt, 10)).unsqueeze(0).to(device)
                return model(st, goal_tensor).item()
                
        def classical_h(s, bt):
            return manhattan_distance_heuristic(s, bt)

        # 1. Classical A*
        c_sol, c_exp, _, c_time = run_search(state, box_targets, classical_h, alg="astar", max_expansions=8000, max_time=4.0)
        if c_sol:
            results["Classical A*"]["solved"] += 1
            results["Classical A*"]["exp"].append(c_exp)
            results["Classical A*"]["time"].append(c_time)

        # 2. Learned A* (Paper model)
        la_sol, la_exp, _, la_time = run_search(state, box_targets, neural_h, alg="astar", max_expansions=8000, max_time=4.0)
        if la_sol:
            results["Paper Learned A*"]["solved"] += 1
            results["Paper Learned A*"]["exp"].append(la_exp)
            results["Paper Learned A*"]["time"].append(la_time)

        # 3. Learned GBFS (Paper model)
        lg_sol, lg_exp, _, lg_time = run_search(state, box_targets, neural_h, alg="gbfs", max_expansions=8000, max_time=4.0)
        if lg_sol:
            results["Paper Learned GBFS"]["solved"] += 1
            results["Paper Learned GBFS"]["exp"].append(lg_exp)
            results["Paper Learned GBFS"]["time"].append(lg_time)

        print(f"Map {i:02d}/{n_samples} | Classical A*: {'SOLVED' if c_sol else 'TIMEOUT'} | Learned A*: {'SOLVED' if la_sol else 'TIMEOUT'} | Learned GBFS: {'SOLVED' if lg_sol else 'TIMEOUT'}", flush=True)

    print("\n==================================================", flush=True)
    print("             OFFICIAL BENCHMARK SUMMARY           ", flush=True)
    print("==================================================", flush=True)
    for k, v in results.items():
        rate = v["solved"] / n_samples * 100
        mean_exp = np.mean(v["exp"]) if v["exp"] else 0
        med_exp = np.median(v["exp"]) if v["exp"] else 0
        mean_t = np.mean(v["time"])*1000 if v["time"] else 0
        print(f"[{k}]", flush=True)
        print(f"  • Solved: {v['solved']} / {n_samples} ({rate:.1f}%)", flush=True)
        print(f"  • Expansions (Mean / Median): {mean_exp:.1f} / {med_exp:.1f}", flush=True)
        print(f"  • Mean Search Time: {mean_t:.1f} ms", flush=True)
    print("==================================================\n", flush=True)

if __name__ == "__main__":
    test_file = os.path.join("Optimize-Planning-Heuristics-to-Rank", "sokoban", "test", "states10test.txt")
    evaluate_full(test_file, checkpoint_path="finalSok3_pytorch.pt", n_samples=25)
