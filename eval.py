"""
Greedy evaluation (ε=0) over N episodes. Report success rate & average return.
Meets: requested evaluation metric + map difficulty comparisons. :contentReference[oaicite:9]{index=9}
"""
from pathlib import Path
import numpy as np
from env import make_env
from utils import load_checkpoint

def evaluate_checkpoint(cfg, checkpoint_path: Path, episodes: int):
    env = make_env(cfg.env_map, cfg.seed + 999)
    Q = load_checkpoint(checkpoint_path)
    successes, returns = [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=cfg.seed + 5000 + ep)
        done = False
        ret = 0.0
        while not done:
            a = int(np.argmax(Q[s]))
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ret += r
        successes.append(int(ret > 0.0))
        returns.append(ret)
    env.close()
    return {
        "episodes": episodes,
        "success_rate": float(np.mean(successes)),
        "avg_return": float(np.mean(returns)),
    }
