import time
import heapq
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.sliding_tile.env import (
    get_goal_board,
    manhattan_linear_conflict,
    get_valid_neighbors,
    encode_boards_25ch,
)


def solve_sliding_tile_astar_strict(
    init_board: np.ndarray,
    dim: int,
    mode: str,
    model: nn.Module,
    device: torch.device,
    timeout: float = 360.0,
    max_expansions: Optional[int] = None,
    kappa: float = 0.50,
    beta: float = 2.5,
    lambda_min: float = 0.75,
    mc_samples: int = 10,
    mc_dropout_p: float = 0.20,
) -> Dict:
    """
    Strict implementation of A* matching Optimize-Planning-Heuristics-to-Rank/slidingtile/test/test_on_Astar.py:
      - Exact raw neural output h(s) from the pre-trained `stile` model (with h(goal) = 0).
      - Exact openSet and closedSet with closed-state reopening when g_new < g_old.
      - No artificial expansion cap by default (max_expansions=None), running up to `timeout` (360s = 6 min).
      - Records both `paper_expansions` (the exact `states_expanded` counter in test_on_Astar.py lines 108 & 123)
        and `nodes_popped` (standard A* popped expansions).
      - Modes supported on the pre-trained `stile` model:
          1. 'paper_baseline_lstar': Exact paper A* with h(s) = h_stile(s)
          2. 'mc_dropout_risk_averse': Risk-Averse A* with h(s) = h_stile(s) + kappa * sigma_MC(s) (AGENTS.md Rule 3)
          3. 'mc_dropout_conf_aware': Confidence-Aware A* blending h_stile(s) and scale-matched MD+LC
    """
    t0 = time.time()
    goal = get_goal_board(dim)
    goal_bytes = goal.tobytes()
    init_bytes = init_board.tobytes()

    if init_bytes == goal_bytes:
        return {
            "solved": True,
            "paper_expansions": 0,
            "nodes_popped": 0,
            "plan_length": 0,
            "runtime": 0.0,
            "mean_sigma": 0.0,
            "mean_lambda": 1.0,
        }

    goal_t = encode_boards_25ch([goal], device)
    init_md_lc = max(1.0, manhattan_linear_conflict(init_board, dim))

    # Evaluate initial state to establish scale ratio between h_stile and MD+LC for conf_aware mode
    init_t = encode_boards_25ch([init_board], device)
    with torch.no_grad():
        h_init_raw = float(model(init_t, goal_t, mc_dropout_p=0.0)[0].item())
    md_scale_ratio = max(1.0, abs(h_init_raw) / init_md_lc)
    sigma_norm_denom = max(abs(h_init_raw) * 0.12, 1.0)

    sigmas_logged: List[float] = []
    lambdas_logged: List[float] = []

    def evaluate_children(boards: List[np.ndarray]) -> np.ndarray:
        B = len(boards)
        if B == 0:
            return np.zeros(0, dtype=np.float64)

        # Check if any board is the goal state (test_on_Astar.py line 55: if check_goal: return 0)
        is_goal_mask = np.array([b.tobytes() == goal_bytes for b in boards], dtype=bool)

        if mode == "classical_md_lc":
            vals = np.array([manhattan_linear_conflict(b, dim) for b in boards], dtype=np.float64)
            vals[is_goal_mask] = 0.0
            return vals

        s_t = encode_boards_25ch(boards, device)
        gb_t = goal_t.expand(B, -1, -1, -1)

        with torch.no_grad():
            raw_det = model(s_t, gb_t, mc_dropout_p=0.0).cpu().numpy().astype(np.float64)

            if mode == "paper_baseline_lstar":
                raw_det[is_goal_mask] = 0.0
                return raw_det

            # MC-Dropout passes on the pre-trained stile model
            s_rep = s_t.repeat(mc_samples, 1, 1, 1)
            g_rep = gb_t.repeat(mc_samples, 1, 1, 1)
            raw_mc = model(s_rep, g_rep, mc_dropout_p=mc_dropout_p).view(mc_samples, B).cpu().numpy().astype(np.float64)

            # Zero-center each MC pass across the batch when B > 1 (AGENTS.md Rule 1)
            if B > 1:
                pass_bias = np.mean(raw_mc, axis=1, keepdims=True) - np.mean(raw_mc)
                zc_mc = raw_mc - pass_bias
            else:
                zc_mc = raw_mc

            sigma = np.std(zc_mc, axis=0)
            sigma[is_goal_mask] = 0.0
            sigmas_logged.extend(sigma.tolist())

            if mode == "mc_dropout_risk_averse":
                res = raw_det + kappa * sigma
                res[is_goal_mask] = 0.0
                return res

            elif mode == "mc_dropout_conf_aware":
                md_scaled = np.array(
                    [md_scale_ratio * manhattan_linear_conflict(b, dim) for b in boards],
                    dtype=np.float64,
                )
                rel_sigma = sigma / sigma_norm_denom
                lam = np.maximum(lambda_min, np.exp(-beta * rel_sigma))
                lam[is_goal_mask] = 1.0
                lambdas_logged.extend(lam.tolist())
                res = lam * raw_det + (1.0 - lam) * md_scaled
                res[is_goal_mask] = 0.0
                return res

            else:
                raise ValueError(f"Unknown mode: {mode}")

    h0 = float(evaluate_children([init_board])[0])

    # Match test_on_Astar.py data structures:
    # Entry in openSet / closedSet: [g, h, f, parent_bytes, act_no, board_ref]
    openSet: Dict[bytes, list] = {
        init_bytes: [0, h0, h0, None, 0, init_board]
    }
    closedSet: Dict[bytes, list] = {}
    heap: List[Tuple[float, int, bytes]] = []
    tie_counter = 0

    state_bytes = init_bytes
    state_entry = openSet[state_bytes]
    paper_states_expanded = 0
    nodes_popped = 0

    while True:
        elapsed = time.time() - t0
        if elapsed > timeout:
            return {
                "solved": False,
                "paper_expansions": paper_states_expanded,
                "nodes_popped": nodes_popped,
                "plan_length": -1,
                "runtime": elapsed,
                "mean_sigma": float(np.mean(sigmas_logged)) if sigmas_logged else 0.0,
                "mean_lambda": float(np.mean(lambdas_logged)) if lambdas_logged else 1.0,
            }

        if max_expansions is not None and nodes_popped >= max_expansions:
            return {
                "solved": False,
                "paper_expansions": paper_states_expanded,
                "nodes_popped": nodes_popped,
                "plan_length": -1,
                "runtime": elapsed,
                "mean_sigma": float(np.mean(sigmas_logged)) if sigmas_logged else 0.0,
                "mean_lambda": float(np.mean(lambdas_logged)) if lambdas_logged else 1.0,
            }

        # Check goal on popped state (test_on_Astar.py line 83)
        if state_bytes == goal_bytes:
            closedSet[state_bytes] = state_entry
            plan_len = 0
            cur_b = state_bytes
            while cur_b != init_bytes:
                plan_len += 1
                cur_b = closedSet[cur_b][3]
            return {
                "solved": True,
                "paper_expansions": paper_states_expanded,
                "nodes_popped": nodes_popped,
                "plan_length": plan_len,
                "runtime": time.time() - t0,
                "mean_sigma": float(np.mean(sigmas_logged)) if sigmas_logged else 0.0,
                "mean_lambda": float(np.mean(lambdas_logged)) if lambdas_logged else 1.0,
            }

        closedSet[state_bytes] = state_entry
        nodes_popped += 1
        g_curr = state_entry[0]
        board_curr = state_entry[5]

        neighbors = get_valid_neighbors(board_curr, dim)
        # Identify brand-new states that need neural heuristic evaluation
        new_indices = []
        new_boards = []
        for idx_n, (nb, act_id) in enumerate(neighbors):
            nb_bytes = nb.tobytes()
            if nb_bytes not in closedSet and nb_bytes not in openSet:
                new_indices.append(idx_n)
                new_boards.append(nb)

        new_h_map = {}
        if new_boards:
            h_vals = evaluate_children(new_boards)
            for idx_n, hv in zip(new_indices, h_vals):
                new_h_map[idx_n] = float(hv)

        # Process all neighbors exactly as in test_on_Astar.py lines 96-124
        for idx_n, (nb, act_id) in enumerate(neighbors):
            nb_bytes = nb.tobytes()
            new_g = g_curr + 1

            if nb_bytes in closedSet:
                if closedSet[nb_bytes][0] > new_g:  # Reopening closed state (line 101)
                    closedSet[nb_bytes][0] = new_g
                    closedSet[nb_bytes][2] = new_g + closedSet[nb_bytes][1]
                    closedSet[nb_bytes][3] = state_bytes
                    closedSet[nb_bytes][4] = act_id
                    openSet[nb_bytes] = closedSet[nb_bytes]
                    tie_counter += 1
                    heapq.heappush(heap, (closedSet[nb_bytes][2], tie_counter, nb_bytes))
                    paper_states_expanded += 1

            elif nb_bytes in openSet:
                if openSet[nb_bytes][0] > new_g:  # Updating open state (line 111)
                    openSet[nb_bytes][0] = new_g
                    openSet[nb_bytes][2] = new_g + openSet[nb_bytes][1]
                    openSet[nb_bytes][3] = state_bytes
                    openSet[nb_bytes][4] = act_id
                    tie_counter += 1
                    heapq.heappush(heap, (openSet[nb_bytes][2], tie_counter, nb_bytes))

            else:
                h_val = new_h_map[idx_n]
                f_val = new_g + h_val
                openSet[nb_bytes] = [new_g, h_val, f_val, state_bytes, act_id, nb]
                tie_counter += 1
                heapq.heappush(heap, (f_val, tie_counter, nb_bytes))
                paper_states_expanded += 1

        if not heap:
            return {
                "solved": False,
                "paper_expansions": paper_states_expanded,
                "nodes_popped": nodes_popped,
                "plan_length": -1,
                "runtime": time.time() - t0,
                "mean_sigma": float(np.mean(sigmas_logged)) if sigmas_logged else 0.0,
                "mean_lambda": float(np.mean(lambdas_logged)) if lambdas_logged else 1.0,
            }

        _, _, state_bytes = heapq.heappop(heap)
        state_entry = openSet[state_bytes]


# Alias for backward compatibility
solve_sliding_tile_astar = solve_sliding_tile_astar_strict
