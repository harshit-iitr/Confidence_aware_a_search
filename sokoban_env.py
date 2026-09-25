import numpy as np
import torch
from typing import List, Tuple, Optional

# Grid values:
# 0 = Wall
# 1 = Empty space
# 2 = Box target
# 3 = Player (Sokoban)
# 4 = Box

class SokobanEnv:
    def __init__(self, dim: int = 10):
        self.dim = dim

    @staticmethod
    def load_dataset(file_path: str) -> List[np.ndarray]:
        """Loads non-duplicate 2D state arrays from dataset text file."""
        states = []
        seen = set()
        with open(file_path, "r") as f:
            for line in f:
                nums = tuple(int(x) for x in line.split())
                if nums and nums not in seen:
                    seen.add(nums)
                    arr = np.array(nums, dtype=np.int32).reshape(10, 10)
                    states.append(arr)
        return states

    @staticmethod
    def get_box_targets(state: np.ndarray) -> List[Tuple[int, int]]:
        """Returns list of (row, col) coordinates for box targets (2)."""
        rows, cols = np.where(state == 2)
        return list(zip(rows.tolist(), cols.tolist()))

    @staticmethod
    def get_goal_state(init_state: np.ndarray, box_targets: List[Tuple[int, int]]) -> np.ndarray:
        """Constructs canonical goal state where all targets have boxes (4)."""
        goal = init_state.copy()
        # Clear player (3) and boxes (4) to empty (1)
        goal[(goal == 3) | (goal == 4)] = 1
        # Place boxes (4) on all targets
        for r, c in box_targets:
            goal[r, c] = 4
        return goal

    @staticmethod
    def is_goal(state: np.ndarray, box_targets: List[Tuple[int, int]]) -> bool:
        """Returns True if every box target contains a box (4)."""
        for r, c in box_targets:
            if state[r, c] != 4:
                return False
        return True

    @staticmethod
    def get_neighbors(state: np.ndarray, box_targets: List[Tuple[int, int]], dim: int = 10):
        """
        Generates valid successor states, action IDs, and step costs (1).
        Actions: 4 directions (Down=6, Up=5, Right=4, Left=3 for push; or standard).
        """
        player_pos = np.where(state == 3)
        if len(player_pos[0]) == 0:
            return [], [], []
        pr, pc = int(player_pos[0][0]), int(player_pos[1][0])
        
        target_set = set(box_targets)
        directions = [
            (1, 0, 6, 2),   # Down: (dr, dc, move_act, push_act)
            (-1, 0, 5, 1),  # Up
            (0, 1, 4, 4),   # Right
            (0, -1, 3, 3)   # Left
        ]
        
        next_states = []
        actions = []
        costs = []

        for dr, dc, move_act, push_act in directions:
            nr, nc = pr + dr, pc + dc
            if not (0 <= nr < dim and 0 <= nc < dim):
                continue
            
            # Simple movement to floor or target
            if state[nr, nc] in (1, 2):
                new_state = state.copy()
                new_state[pr, pc] = 2 if (pr, pc) in target_set else 1
                new_state[nr, nc] = 3
                next_states.append(new_state)
                actions.append(move_act)
                costs.append(1)

            # Pushing a box
            elif state[nr, nc] == 4:
                nnr, nnc = nr + dr, nc + dc
                if 0 <= nnr < dim and 0 <= nnc < dim and state[nnr, nnc] in (1, 2):
                    new_state = state.copy()
                    new_state[pr, pc] = 2 if (pr, pc) in target_set else 1
                    new_state[nr, nc] = 3
                    new_state[nnr, nnc] = 4
                    next_states.append(new_state)
                    actions.append(push_act)
                    costs.append(1)

        return next_states, actions, costs

    @staticmethod
    def state_to_tensor(state: np.ndarray, box_targets: List[Tuple[int, int]], dim: int = 10) -> np.ndarray:
        """
        Converts 2D state into 5-channel one-hot tensor of shape (5, H, W).
        Channels: 0=wall, 1=floor, 2=target, 3=player, 4=box.
        Ensures target channel is 1 at target locations even when occupied.
        """
        tensor = np.zeros((5, dim, dim), dtype=np.float32)
        for c in range(5):
            tensor[c] = (state == c).astype(np.float32)
        for r, c in box_targets:
            tensor[2, r, c] = 1.0
        return tensor

    @staticmethod
    def state_to_key(state: np.ndarray) -> bytes:
        return state.tobytes()
