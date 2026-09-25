import os
import time
import numpy as np
import torch
import heapq
from typing import List

from torch_model import ChrestienHeuristicNet
from sokoban_env import SokobanEnv

def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def author_to_tensor(state: np.ndarray, box_targets: List, dim: int = 10) -> np.ndarray:
    """Exact replication of authors' to_categorical_tensor."""
    box_rows, box_cols = np.where(state == 4)
    pos = list(zip(box_rows.tolist(), box_cols.tolist()))
    
    tensor = np.zeros((5, dim, dim), dtype=np.float32)
    for c in range(5):
        tensor[c] = (state == c).astype(np.float32)
    for r, c in box_targets:
        tensor[2, r, c] = 1.0
    for r, c in pos:
        tensor[2, r, c] = 1.0  # Author's quirk in train_sok.py line 16
    return tensor

def run_gbfs_test(test_file: str, n_instances: int = 50, timeout: float = 15.0):
    device = get_device()
    print(f"Testing GBFS with Paper weights on {n_instances} maps (Timeout: {timeout}s)...", flush=True)
    
    states = SokobanEnv.load_dataset(test_file)[:n_instances]
    model = ChrestienHeuristicNet(dim=10).to(device)
    model.load_state_dict(torch.load("finalSok3_pytorch.pt", map_location=device))
    model.eval()

    solved = 0
    expansions_list = []
    
    for i, state in enumerate(states, 1):
        box_targets = SokobanEnv.get_box_targets(state)
        goal_state = SokobanEnv.get_goal_state(state, box_targets)
        
        with torch.no_grad():
            goal_t = torch.from_numpy(author_to_tensor(goal_state, box_targets, 10)).unsqueeze(0).to(device)
            
        def h_fn(s, bt):
            with torch.no_grad():
                st = torch.from_numpy(author_to_tensor(s, bt, 10)).unsqueeze(0).to(device)
                return model(st, goal_t).item()

        start_t = time.time()
        start_h = h_fn(state, box_targets)
        
        # Priority queue with (h, counter, state)
        open_heap = [(start_h, 0, state, 0)]
        closed_set = set()
        counter = 0
        exp = 0
        is_sol = False
        
        while open_heap:
            if (time.time() - start_t) > timeout or exp > 25000:
                break
            h_val, _, curr_state, g_val = heapq.heappop(open_heap)
            k = curr_state.tobytes()
            if k in closed_set:
                continue
            closed_set.add(k)
            exp += 1
            
            if SokobanEnv.is_goal(curr_state, box_targets):
                is_sol = True
                break
                
            next_states, _, costs = SokobanEnv.get_neighbors(curr_state, box_targets, 10)
            for ns, c in zip(next_states, costs):
                nk = ns.tobytes()
                if nk in closed_set:
                    continue
                nh = h_fn(ns, box_targets)
                counter += 1
                heapq.heappush(open_heap, (nh, counter, ns, g_val + c))
                
        if is_sol:
            solved += 1
            expansions_list.append(exp)
            print(f"Map {i:02d}/{n_instances} | SOLVED in {exp} expansions ({time.time()-start_t:.2f}s)", flush=True)
        else:
            print(f"Map {i:02d}/{n_instances} | TIMEOUT/FAIL after {exp} expansions ({time.time()-start_t:.2f}s)", flush=True)

    print(f"\n==================================================", flush=True)
    print(f"TOTAL SOLVED: {solved} / {n_instances} ({solved/n_instances*100:.1f}%)", flush=True)
    if expansions_list:
        print(f"Median Expansions: {np.median(expansions_list):.1f}", flush=True)
        print(f"Mean Expansions: {np.mean(expansions_list):.1f}", flush=True)
    print(f"==================================================", flush=True)

if __name__ == "__main__":
    test_file = os.path.join("Optimize-Planning-Heuristics-to-Rank", "sokoban", "test", "states10test.txt")
    run_gbfs_test(test_file, n_instances=30, timeout=10.0)
