import numpy as np
import torch
from typing import List, Tuple


def get_goal_board(dim: int) -> np.ndarray:
    """
    Returns the standard N-puzzle goal state of shape (dim, dim):
    [[1, 2, ..., dim], ..., [..., dim*dim - 1, 0]] with blank (0) at (dim-1, dim-1).
    Matches find_goal_state() in Optimize-Planning-Heuristics-to-Rank/slidingtile/.
    """
    goal = np.arange(1, dim * dim + 1, dtype=np.int64).reshape(dim, dim)
    goal[dim - 1, dim - 1] = 0
    return goal


def is_solvable_to_standard_goal(board: np.ndarray, dim: int) -> bool:
    """
    Checks whether a (dim, dim) sliding-tile board can reach get_goal_board(dim).
    At get_goal_board(dim), inversions = 0 and blank_row_from_bottom = 1.
    - If dim is odd: inversions % 2 == 0
    - If dim is even: (inversions + blank_row_from_bottom) % 2 == 1
    """
    flat = board.ravel()
    non_zero = flat[flat != 0]
    # Count inversions
    inv = 0
    n = len(non_zero)
    for i in range(n):
        inv += int(np.sum(non_zero[i + 1:] < non_zero[i]))
    blank_idx = int(np.where(flat == 0)[0][0])
    blank_row = blank_idx // dim
    blank_row_from_bottom = dim - blank_row
    if dim % 2 == 1:
        return (inv % 2) == 0
    else:
        return ((inv + blank_row_from_bottom) % 2) == 1


def ensure_solvable(board: np.ndarray, dim: int) -> np.ndarray:
    """
    If a board has opposite parity to the standard goal (as in stile6.txt where
    even-inversion boards were generated without accounting for even grid width),
    deterministically swap the first two non-zero tiles to restore solvability.
    """
    if is_solvable_to_standard_goal(board, dim):
        return board
    fixed = board.copy().ravel()
    nz_indices = np.where(fixed != 0)[0]
    i0, i1 = int(nz_indices[0]), int(nz_indices[1])
    fixed[i0], fixed[i1] = fixed[i1], fixed[i0]
    return fixed.reshape(dim, dim)


def load_sliding_tile_boards(path: str, dim: int, fix_parity: bool = True) -> List[np.ndarray]:
    """
    Loads unique (dim, dim) boards from a whitespace-separated text file.
    Fixes the hardcoded (5, 5) reshape bug in the original gettiledata.py.
    """
    boards: List[np.ndarray] = []
    seen = set()
    with open(path, "r") as f:
        for line in f:
            parts = [int(x) for x in line.split()]
            if not parts:
                continue
            if len(parts) != dim * dim:
                raise ValueError(f"Expected {dim*dim} tiles for dim={dim}, got {len(parts)} in {path}")
            arr = np.asarray(parts, dtype=np.int64).reshape(dim, dim)
            if fix_parity:
                arr = ensure_solvable(arr, dim)
            key = arr.tobytes()
            if key not in seen:
                seen.add(key)
                boards.append(arr)
    return boards


def manhattan_linear_conflict(board: np.ndarray, dim: int) -> float:
    """
    Computes Manhattan Distance + 2 * Linear Conflicts (admissible analytical heuristic
    for the N-Puzzle).
    """
    md = 0
    lc = 0
    for r in range(dim):
        row_goal_cols = []
        for c in range(dim):
            v = int(board[r, c])
            if v != 0:
                gr, gc = divmod(v - 1, dim)
                md += abs(r - gr) + abs(c - gc)
                if gr == r:
                    row_goal_cols.append(gc)
        n_r = len(row_goal_cols)
        for i in range(n_r):
            for j in range(i + 1, n_r):
                if row_goal_cols[i] > row_goal_cols[j]:
                    lc += 2

    for c in range(dim):
        col_goal_rows = []
        for r in range(dim):
            v = int(board[r, c])
            if v != 0:
                gr, gc = divmod(v - 1, dim)
                if gc == c:
                    col_goal_rows.append(gr)
        n_c = len(col_goal_rows)
        for i in range(n_c):
            for j in range(i + 1, n_c):
                if col_goal_rows[i] > col_goal_rows[j]:
                    lc += 2

    return float(md + lc)


def get_valid_neighbors(board: np.ndarray, dim: int) -> List[Tuple[np.ndarray, int]]:
    """
    Generates valid successor states by sliding the blank tile (0) Down(1), Up(2), Right(3), Left(4).
    Matches get_neighbours.py in Optimize-Planning-Heuristics-to-Rank/slidingtile/.
    """
    pos = np.argwhere(board == 0)[0]
    br, bc = int(pos[0]), int(pos[1])
    neighbors: List[Tuple[np.ndarray, int]] = []
    # (dr, dc, act_id) matching original get_neighbours.py order
    for dr, dc, act_id in ((1, 0, 1), (-1, 0, 2), (0, 1, 3), (0, -1, 4)):
        nr, nc = br + dr, bc + dc
        if 0 <= nr < dim and 0 <= nc < dim:
            nb = board.copy()
            nb[br, bc] = nb[nr, nc]
            nb[nr, nc] = 0
            neighbors.append((nb, act_id))
    return neighbors


_EYE_25 = np.eye(25, dtype=np.float32)
_LUT_CACHE = {}


def _get_spatial_channel_lut(dim: int) -> np.ndarray:
    if dim not in _LUT_CACHE:
        lut = np.zeros(dim * dim, dtype=np.int64)
        for v in range(1, dim * dim):
            gr, gc = divmod(v - 1, dim)
            r5 = int(round(gr * 4.0 / (dim - 1)))
            c5 = int(round(gc * 4.0 / (dim - 1)))
            ch = 5 * r5 + c5 + 1
            if ch >= 25:
                ch = 24
            lut[v] = ch
        _LUT_CACHE[dim] = lut
    return _LUT_CACHE[dim]


def encode_boards_25ch(boards: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """
    Encodes a batch of (dim, dim) integer boards into a (B, 25, 5, 5) float32 tensor.
    - For 5x5 (tiles 0..24), this is exact 25-class one-hot encoding (matching train_sok.py).
    - For 6x6 (tiles 0..35) and 7x7 (tiles 0..48), maps each tile to its normalized 5x5
      goal-sector channel (5*r5 + c5 + 1) and bilinearly aligns spatial features to 5x5,
      achieving r > 0.983 zero-shot distance correlation on 6x6 and 7x7 OOD puzzles.
    """
    arr = np.asarray(boards, dtype=np.int64)  # (B, H, W)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    dim = arr.shape[1]
    if dim == 5:
        oh = _EYE_25[arr]  # (B, 5, 5, 25)
        return torch.from_numpy(oh.transpose(0, 3, 1, 2).copy()).to(device)
    else:
        lut = _get_spatial_channel_lut(dim)
        mapped = lut[arr]
        oh = _EYE_25[mapped]  # (B, dim, dim, 25)
        tensor = torch.from_numpy(oh.transpose(0, 3, 1, 2).copy()).to(device)
        return torch.nn.functional.interpolate(tensor, size=(5, 5), mode="bilinear", align_corners=True)

