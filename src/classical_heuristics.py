import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment

def manhattan_distance_heuristic(state: np.ndarray, box_targets: List[Tuple[int, int]]) -> float:
    """
    Computes an admissible classical heuristic for Sokoban using Minimum Weight 
    Bipartite Matching (Hungarian algorithm) between current box positions and targets.
    """
    box_rows, box_cols = np.where(state == 4)
    boxes = list(zip(box_rows.tolist(), box_cols.tolist()))
    
    if not boxes or not box_targets:
        return 0.0

    n_boxes = len(boxes)
    n_targets = len(box_targets)
    
    # Cost matrix of Manhattan distances
    cost_matrix = np.zeros((n_boxes, n_targets), dtype=np.float32)
    for i, (br, bc) in enumerate(boxes):
        for j, (tr, tc) in enumerate(box_targets):
            cost_matrix[i, j] = abs(br - tr) + abs(bc - tc)
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_dist = float(cost_matrix[row_ind, col_ind].sum())
    return total_dist


def is_deadlock(state: np.ndarray, box_targets: List[Tuple[int, int]], dim: int = 10) -> bool:
    """
    Detects provable permanent deadlocks in Sokoban:
    1. Corner deadlocks (box stuck in orthogonal corner not on target).
    2. 2x2 block deadlocks (2x2 square of boxes and walls with at least one non-target box).
    3. Wall-pair deadlocks (two adjacent boxes against a wall where at least one is non-target).
    """
    box_rows, box_cols = np.where(state == 4)
    boxes = list(zip(box_rows.tolist(), box_cols.tolist()))
    if not boxes:
        return False
        
    target_set = set(box_targets)
    box_set = set(boxes)
    
    # 1. Corner Deadlocks
    for r, c in boxes:
        if (r, c) not in target_set:
            up = (r == 0) or (state[r-1, c] == 0)
            down = (r == dim - 1) or (state[r+1, c] == 0)
            left = (c == 0) or (state[r, c-1] == 0)
            right = (c == dim - 1) or (state[r, c+1] == 0)
            if (up or down) and (left or right):
                return True

    # 2. 2x2 Block Deadlocks
    for r in range(dim - 1):
        for c in range(dim - 1):
            quad = [(r, c), (r+1, c), (r, c+1), (r+1, c+1)]
            has_box = False
            has_non_target_box = False
            all_blocked = True
            for qr, qc in quad:
                val = state[qr, qc]
                if val == 4:
                    has_box = True
                    if (qr, qc) not in target_set:
                        has_non_target_box = True
                elif val != 0:
                    all_blocked = False
                    break
            if all_blocked and has_box and has_non_target_box:
                return True

    # 3. Two adjacent boxes along a wall
    for r, c in boxes:
        # Horizontal pair
        if (r, c+1) in box_set:
            if (r, c) not in target_set or (r, c+1) not in target_set:
                wall_up = (r == 0) or (state[r-1, c] == 0 and state[r-1, c+1] == 0)
                wall_down = (r == dim - 1) or (state[r+1, c] == 0 and state[r+1, c+1] == 0)
                if wall_up or wall_down:
                    return True
        # Vertical pair
        if (r+1, c) in box_set:
            if (r, c) not in target_set or (r+1, c) not in target_set:
                wall_left = (c == 0) or (state[r, c-1] == 0 and state[r+1, c-1] == 0)
                wall_right = (c == dim - 1) or (state[r, c+1] == 0 and state[r+1, c+1] == 0)
                if wall_left or wall_right:
                    return True

    return False


def deadlock_aware_heuristic(state: np.ndarray, box_targets: List[Tuple[int, int]], dim: int = 10) -> float:
    """
    Computes deadlock-aware heuristic: returns 1e6 if state is in a permanent deadlock,
    otherwise returns Hungarian Manhattan distance.
    """
    if is_deadlock(state, box_targets, dim):
        return 1e6
    return manhattan_distance_heuristic(state, box_targets)

