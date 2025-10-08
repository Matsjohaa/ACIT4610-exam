"""
Baselines required by the brief: random policy and a simple heuristic path attempt.
Heuristic: computes a shortest path on a deterministic grid (ignoring slippage) and follows it,
demonstrating failure under stochastic transitions. :contentReference[oaicite:10]{index=10}
"""
import numpy as np
from env import make_env

ACTIONS = [0,1,2,3]  # Left, Down, Right, Up (FrozenLake-v1)

def run_random_baseline(cfg, episodes: int):
    env = make_env(cfg.env_map, cfg.seed + 111)
    rng = np.random.default_rng(cfg.seed)
    successes = []
    for ep in range(episodes):
        s, _ = env.reset(seed=cfg.seed + ep)
        done = False
        ret = 0.0
        while not done:
            a = rng.integers(len(ACTIONS))
            s, r, term, trunc, _ = env.step(int(a))
            done = term or trunc
            ret += r
        successes.append(int(ret > 0.0))
    env.close()
    return {"baseline": "random", "episodes": episodes, "success_rate": float(np.mean(successes))}

def run_heuristic_baseline(cfg, episodes: int):
    env = make_env(cfg.env_map, cfg.seed + 222)
    # Placeholder: compute a nominal shortest path on the grid ignoring slippage.
    # Then replay that action sequence each episode.
    # You’ll implement a small BFS on the grid desc to obtain a path of actions.
    path_actions = _dummy_nominal_path(env)  # TODO: replace with BFS-based plan
    successes = []
    for ep in range(episodes):
        s, _ = env.reset(seed=cfg.seed + ep)
        done = False
        ret = 0.0
        idx = 0
        while not done:
            a = path_actions[idx % len(path_actions)]
            s, r, term, trunc, _ = env.step(int(a))
            done = term or trunc
            ret += r
            idx += 1
        successes.append(int(ret > 0.0))
    env.close()
    return {"baseline": "heuristic", "episodes": episodes, "success_rate": float(np.mean(successes))}

def _dummy_nominal_path(env):
    # Minimal placeholder; replace with real grid BFS (Left=0, Down=1, Right=2, Up=3).
    return [2,2,1,1,2,1,2,1]
