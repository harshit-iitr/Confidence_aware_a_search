import os
import sys
import numpy as np
from gym_sokoban.envs.room_utils import generate_room

def generate_5box_dataset(n_target: int = 100, output_file: str = "sokoban_5box_test.txt", seed: int = 42):
    np.random.seed(seed)
    print(f"Generating {n_target} unique 10x10 Sokoban mazes with 5 boxes...", flush=True)
    
    unique_states = []
    seen = set()
    attempts = 0
    
    while len(unique_states) < n_target:
        attempts += 1
        try:
            rs, state, _ = generate_room(dim=(10, 10), num_boxes=5, num_steps=30, tries=20)
            # Map player from 5 to 3
            grid = state.copy()
            grid[grid == 5] = 3
            
            # Check conditions:
            # Exactly 5 boxes (4)
            # Exactly 1 player (3)
            # Exactly 5 targets (2) that are unoccupied
            if (grid == 4).sum() != 5:
                continue
            if (grid == 3).sum() != 1:
                continue
            if (grid == 2).sum() != 5:
                continue
                
            flat = tuple(grid.flatten().tolist())
            if flat not in seen:
                seen.add(flat)
                unique_states.append(flat)
                if len(unique_states) % 10 == 0 or len(unique_states) == n_target:
                    print(f"  Generated {len(unique_states)} / {n_target} valid 5-box mazes (attempts: {attempts})", flush=True)
        except Exception:
            continue
            
    with open(output_file, "w", encoding="utf-8") as f:
        for s in unique_states:
            f.write(" ".join(map(str, s)) + "\n")
            
    print(f"Successfully saved {len(unique_states)} instances to '{output_file}'!\n", flush=True)

if __name__ == "__main__":
    n = 100 if len(sys.argv) < 2 else int(sys.argv[1])
    generate_5box_dataset(n_target=n)
