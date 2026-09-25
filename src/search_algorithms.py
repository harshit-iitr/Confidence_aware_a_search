import time
import heapq
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sokoban_env import SokobanEnv

class SearchNode:
    __slots__ = ('state', 'key', 'g', 'h', 'f', 'parent_key', 'action', 'is_on_path')
    def __init__(self, state: np.ndarray, g: float, h: float, parent_key: Optional[bytes] = None, action: int = 0):
        self.state = state
        self.key = SokobanEnv.state_to_key(state)
        self.g = g
        self.h = h
        self.f = g + h
        self.parent_key = parent_key
        self.action = action
        self.is_on_path = False

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        return self.h < other.h


def run_astar(
    init_state: np.ndarray,
    box_targets: List[Tuple[int, int]],
    heuristic_fn: Callable[[np.ndarray, List[Tuple[int, int]]], float],
    max_time: float = 30.0,
    max_expansions: int = 10000,
    dim: int = 10
) -> Tuple[bool, int, float, List[int], float]:
    """
    Standard A* search with timeout and expansion limit.
    """
    start_time = time.time()
    
    init_h = heuristic_fn(init_state, box_targets)
    start_node = SearchNode(init_state, g=0.0, h=init_h, parent_key=None, action=0)
    
    open_heap = []
    heapq.heappush(open_heap, (start_node.f, 0, start_node))
    
    open_dict: Dict[bytes, float] = {start_node.key: start_node.g}
    closed_dict: Dict[bytes, SearchNode] = {}
    
    counter = 0
    states_expanded = 0

    while open_heap:
        if (time.time() - start_time) > max_time or states_expanded >= max_expansions:
            return False, states_expanded, float("inf"), [], time.time() - start_time
            
        _, _, current = heapq.heappop(open_heap)
        if current.key in closed_dict:
            continue
            
        closed_dict[current.key] = current
        states_expanded += 1

        if SokobanEnv.is_goal(current.state, box_targets):
            elapsed = time.time() - start_time
            actions = []
            curr = current
            while curr.parent_key is not None:
                actions.append(curr.action)
                curr = closed_dict[curr.parent_key]
            actions.reverse()
            return True, states_expanded, current.g, actions, elapsed

        next_states, act_nos, costs = SokobanEnv.get_neighbors(current.state, box_targets, dim)
        for n_state, act, cost in zip(next_states, act_nos, costs):
            n_key = SokobanEnv.state_to_key(n_state)
            new_g = current.g + cost
            
            if n_key in closed_dict:
                continue
                
            if n_key not in open_dict or new_g < open_dict[n_key]:
                open_dict[n_key] = new_g
                h_val = heuristic_fn(n_state, box_targets)
                child_node = SearchNode(n_state, g=new_g, h=h_val, parent_key=current.key, action=act)
                counter += 1
                heapq.heappush(open_heap, (child_node.f, counter, child_node))

    elapsed = time.time() - start_time
    return False, states_expanded, float("inf"), [], elapsed


def extract_lstar_training_data(
    init_state: np.ndarray,
    box_targets: List[Tuple[int, int]],
    heuristic_fn: Callable[[np.ndarray, List[Tuple[int, int]]], float],
    max_time: float = 3.0,
    max_expansions: int = 2000,
    dim: int = 10
) -> Optional[Tuple[List[np.ndarray], List[float], List[np.ndarray], List[float]]]:
    """
    Runs fast A* search to generate on-path and off-path training pairs for L* ranking loss.
    """
    start_time = time.time()
    init_h = heuristic_fn(init_state, box_targets)
    start_node = SearchNode(init_state, g=0.0, h=init_h, parent_key=None, action=0)
    
    open_heap = []
    heapq.heappush(open_heap, (start_node.f, 0, start_node))
    
    all_nodes: Dict[bytes, SearchNode] = {start_node.key: start_node}
    closed_dict: Dict[bytes, SearchNode] = {}
    
    counter = 0
    states_expanded = 0
    goal_node = None

    while open_heap:
        if (time.time() - start_time) > max_time or states_expanded >= max_expansions:
            return None
            
        _, _, current = heapq.heappop(open_heap)
        if current.key in closed_dict:
            continue
        closed_dict[current.key] = current
        states_expanded += 1

        if SokobanEnv.is_goal(current.state, box_targets):
            goal_node = current
            break

        next_states, act_nos, costs = SokobanEnv.get_neighbors(current.state, box_targets, dim)
        for n_state, act, cost in zip(next_states, act_nos, costs):
            n_key = SokobanEnv.state_to_key(n_state)
            new_g = current.g + cost
            
            if n_key in closed_dict:
                continue
                
            if n_key not in all_nodes or new_g < all_nodes[n_key].g:
                h_val = heuristic_fn(n_state, box_targets)
                child_node = SearchNode(n_state, g=new_g, h=h_val, parent_key=current.key, action=act)
                all_nodes[n_key] = child_node
                counter += 1
                heapq.heappush(open_heap, (child_node.f, counter, child_node))

    if goal_node is None:
        return None

    # Mark on-path nodes
    curr = goal_node
    while curr is not None:
        curr.is_on_path = True
        curr = closed_dict.get(curr.parent_key) if curr.parent_key else None

    on_path_states = []
    on_path_g = []
    off_path_states = []
    off_path_g = []

    for node in all_nodes.values():
        if node.is_on_path:
            on_path_states.append(node.state)
            on_path_g.append(node.g)
        else:
            off_path_states.append(node.state)
            off_path_g.append(node.g)

    if not on_path_states or not off_path_states:
        return None

    return on_path_states, on_path_g, off_path_states, off_path_g
