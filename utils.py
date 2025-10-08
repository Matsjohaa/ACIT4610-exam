from pathlib import Path
import json
import numpy as np

def ensure_dirs(*paths: Path):
    for p in paths: p.mkdir(parents=True, exist_ok=True)

def set_global_seeds(seed: int):
    np.random.seed(seed)

def save_checkpoint(Q, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, Q=Q)

def load_checkpoint(path: Path):
    return np.load(path)["Q"]

def save_manifest(cfg, run_dir: Path):
    manifest = {
        "env_map": cfg.env_map,
        "episodes": cfg.episodes,
        "epsilon_schedule": cfg.epsilon_schedule,
        "alpha": cfg.alpha,
        "gamma": cfg.gamma,
        "seed": cfg.seed,
        "run_dir": str(cfg.run_dir),
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

class RunningStats:
    """Online mean/variance (Welford). Call update(x) each episode."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # sum of squares of diffs from mean

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 0 else 0.0

    @property
    def std(self) -> float:
        return self.variance ** 0.5
