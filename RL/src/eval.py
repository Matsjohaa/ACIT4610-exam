from pathlib import Path
import numpy as np
from env import make_env
from utils import load_checkpoint

def evaluate_checkpoint(cfg, checkpoint_path: Path, episodes: int):
    """
    Evaluate a saved Q-table (greedy policy, ε=0) on the same layout used in training.

    This function:
      1) Infers the original run directory from the checkpoint path.
      2) Loads the saved map layout `desc.npy` (if present) to reconstruct the same environment.
      3) Builds the FrozenLake env with `slippery=True`, and no step penalty (unshaped eval).
      4) Loads the Q-table and verifies its state dimension matches the environment.
      5) Runs greedy episodes (no learning, no exploration) and reports success metrics.

    Args:
        cfg: Config object with fields like env_map, seed, run_name, etc.
        checkpoint_path (Path): Path to the saved Q-table (npz with key "Q").
        episodes (int): Number of evaluation episodes to run (bigger = tighter CI).

    Returns:
        dict: A summary with keys:
              - 'run_name': run identifier (from cfg or inferred from path)
              - 'map': map size ('4x4', '6x6', '8x8')
              - 'episodes': number of eval episodes
              - 'success_rate': fraction of episodes that reached the goal
              - 'avg_return': mean per-episode return (equals success_rate for 0/1 rewards)
              - 'seed': base seed (for reproducibility)
              - 'checkpoint': string path to the evaluated checkpoint
              - 'step_penalty': 0.0 (eval is unshaped)
              - 'desc_loaded': whether desc.npy was found/used to fix the layout
    """
    # Locate the run dir from the checkpoint path (…/runs/<run_name>/checkpoints/qtable-final.npz)
    run_dir = checkpoint_path.parent.parent
    desc_path = run_dir / "desc.npy"
    desc = np.load(desc_path) if desc_path.exists() else None

    # Build env USING THE SAME LAYOUT; eval unshaped (step_penalty=0.0)
    env = make_env(cfg.env_map, cfg.seed + 999, step_penalty=0.0, desc=desc)

    Q = load_checkpoint(checkpoint_path)
    n_states = env.observation_space.n

    # Guard against mismatched layouts (e.g., evaluating a 6x6 Q on an 8x8 env)
    if Q.shape[0] != n_states:
        raise ValueError(
            f"Q-table has {Q.shape[0]} states but env has {n_states}. "
            f"Pass --map to match training, and ensure desc.npy exists."
        )

    successes, returns = [], []

    # Greedy evaluation (ε = 0): always pick argmax_a Q[s, a]
    for _ in range(episodes):
        s, _ = env.reset()
        done, ret = False, 0.0
        while not done:
            a = int(np.argmax(Q[s]))
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ret += r
        successes.append(int(ret > 0.0))
        returns.append(ret)

    env.close()

    stats = {
        "run_name": getattr(cfg, "run_name", run_dir.name),
        "map": cfg.env_map,
        "episodes": episodes,
        "success_rate": float(np.mean(successes)),
        "avg_return": float(np.mean(returns)),
        "seed": cfg.seed,
        "checkpoint": str(checkpoint_path),
        "step_penalty": 0.0,              # eval is unshaped for fair comparison
        "desc_loaded": bool(desc is not None),
    }

    # Optionally persist to Results/ (uncomment to save)
    # results_root = getattr(cfg, "results_root", Path("Results"))
    # results_root.mkdir(parents=True, exist_ok=True)
    # out = results_root / f"rl_eval_{cfg.env_map}_{stats['run_name']}.json"
    # with open(out, "w") as f:
    #     import json; json.dump(stats, f, indent=2)
    return stats
