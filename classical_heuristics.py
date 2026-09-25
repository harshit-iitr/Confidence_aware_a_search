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
