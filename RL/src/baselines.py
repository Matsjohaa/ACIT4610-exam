"""
Baselines for comparison with the learned Q-learning agent.

Two policies are implemented:
1. Random baseline – chooses uniformly random actions.
2. Heuristic baseline – follows the deterministic shortest path (BFS)
   ignoring slippage, replanning each step during execution.

Both produce JSON summaries in Results/ with overall success rate
and average number of steps per episode.
"""

import json
from pathlib import Path
import numpy as np
from env import make_env
from collections import deque
from typing import Optional, Tuple, List

# FrozenLake actions (Gymnasium convention)
A_LEFT, A_DOWN, A_RIGHT, A_UP = 0, 1, 2, 3
ACTIONS = [A_LEFT, A_DOWN, A_RIGHT, A_UP]


def run_random_baseline(cfg, episodes: int):
    """
    Evaluate a completely random policy.

    The agent selects each action uniformly at random at every time step.
    This provides a lower-bound reference for success on the same map.

    Args:
        cfg: configuration object with env_map, seed, results_root, etc.
        episodes (int): number of evaluation episodes.

    Returns:
        dict with baseline name, map, episodes, success_rate, and seed.
    """
    env = make_env(cfg.env_map, cfg.seed + 111, step_penalty=0.0)
    rng = np.random.default_rng(cfg.seed)
    successes, step_counts = [], []

    for _ in range(episodes):
        s, _ = env.reset()
        done, ret, steps = False, 0.0, 0
        while not done:
            a = int(rng.integers(len(ACTIONS)))        # random action
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ret += r
            steps += 1
        successes.append(int(ret > 0.0))
        step_counts.append(steps)
    env.close()

    res = {
        "baseline": "random",
        "map": cfg.env_map,
        "episodes": episodes,
        "success_rate": float(np.mean(successes)),
        "avg_steps": float(np.mean(step_counts)),
        "seed": cfg.seed,
    }

    results_root = getattr(cfg, "results_root", Path("Results"))
    results_root.mkdir(parents=True, exist_ok=True)
    with open(
        results_root
        / f"rl_baseline_random_{cfg.env_map}_{getattr(cfg,'run_name','baseline')}.json",
        "w",
    ) as f:
        json.dump(res, f, indent=2)
    return res


def run_heuristic_baseline(cfg, episodes: int):
    """
    Evaluate a deterministic heuristic policy using BFS replanning.

    The agent repeatedly computes the shortest deterministic path
    (ignoring slippage) from its current cell to the goal 'G' and executes
    the first action of that path.  This mimics a classical planner that
    replans after every slip.

    Args:
        cfg: configuration object with env_map, seed, results_root, etc.
        episodes (int): number of evaluation episodes.

    Returns:
        dict with baseline name, map, episodes, success_rate, avg_steps.
    """
    env = make_env(cfg.env_map, cfg.seed + 222, step_penalty=0.0)

    # Extract static grid from environment description
    desc_bytes = env.unwrapped.desc
    desc = desc_bytes.astype("U1")  # convert bytes → strings

    successes, step_counts = [], []

    for _ in range(episodes):
        s, _ = env.reset()
        done, ret, steps = False, 0.0, 0
        while not done:
            # Convert linear state index to (row, col)
            r_idx, c_idx = divmod(s, env.unwrapped.ncol)
            # Replan: BFS path from current location to G
            act_seq = _bfs_shortest_path_actions(desc, start=(r_idx, c_idx))
            a = act_seq[0] if act_seq else A_RIGHT  # fallback if no path found
            s, r, term, trunc, _ = env.step(int(a))
            done = term or trunc
            ret += r
            steps += 1
        successes.append(int(ret > 0.0))
        step_counts.append(steps)

    env.close()

    res = {
        "baseline": "heuristic_replan",
        "map": cfg.env_map,
        "episodes": episodes,
        "success_rate": float(np.mean(successes)),
        "avg_steps": float(np.mean(step_counts)),
        "seed": cfg.seed,
    }

    results_root = getattr(cfg, "results_root", Path("Results"))
    results_root.mkdir(parents=True, exist_ok=True)
    with open(
        results_root
        / f"rl_baseline_heuristic_{cfg.env_map}_{getattr(cfg,'run_name','baseline')}.json",
        "w",
    ) as f:
        json.dump(res, f, indent=2)
    return res


# --------------------------------------------------------------------------- #
# Helper: Breadth-First Search for deterministic shortest path
# --------------------------------------------------------------------------- #
def _bfs_shortest_path_actions(
    desc: np.ndarray, start: Optional[Tuple[int, int]] = None
) -> List[int]:
    """
    Compute the shortest deterministic path to the goal using BFS.

    Args:
        desc: 2D array (bytes or str) with symbols {'S','F','H','G'}.
        start: optional (row, col) coordinate to start from; if None, use 'S'.

    Returns:
        List[int]: action sequence [0,1,2,3] for (Left, Down, Right, Up)
                   from start to 'G'; empty if no path found.
    """
    grid = desc.astype("U1") if desc.dtype.kind == "S" else desc
    nrows, ncols = grid.shape

    S_pos = tuple(map(int, np.argwhere(grid == "S")[0]))
    G_pos = tuple(map(int, np.argwhere(grid == "G")[0]))
    s_pos = start if start is not None else S_pos
    if s_pos == G_pos:
        return []

    moves = [(0, -1, A_LEFT), (1, 0, A_DOWN), (0, 1, A_RIGHT), (-1, 0, A_UP)]

    def in_bounds(r, c): return 0 <= r < nrows and 0 <= c < ncols
    def passable(r, c):  return grid[r, c] != "H"

    q = deque([s_pos])
    visited = {s_pos: None}
    parent_act = {}
    found = False

    while q:
        r, c = q.popleft()
        if (r, c) == G_pos:
            found = True
            break
        for dr, dc, act in moves:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and passable(nr, nc) and (nr, nc) not in visited:
                visited[(nr, nc)] = (r, c)
                parent_act[(nr, nc)] = act
                q.append((nr, nc))

    if not found:
        return []

    # Reconstruct action sequence from goal back to start
    actions_rev = []
    cur = G_pos
    while cur != s_pos:
        actions_rev.append(parent_act[cur])
        cur = visited[cur]
    return list(reversed(actions_rev))
