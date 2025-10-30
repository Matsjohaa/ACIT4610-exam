"""
Utility helpers for training/evaluation runs.

Responsibilities:
  • Filesystem helpers (create directories, save/load checkpoints, save manifest).
  • Reproducibility (set_global_seeds).
  • Lightweight online statistics (RunningStats: mean/std via Welford).

"""

from pathlib import Path
import json
import numpy as np
from typing import Any


def ensure_dirs(*paths: Path) -> None:
    """
    Create all given directories (recursively) if they do not exist.

    Args:
        *paths: One or more Path objects (directories) to create.
    """
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    """
    Set global RNG seed(s) used by NumPy.

    Notes:
        • Gym/Gymnasium environments should be seeded via env.reset(seed=...)
          for full determinism per episode.
        • Use np.random.default_rng(seed) for per-run RNG objects in code.
    """
    np.random.seed(seed)


def save_checkpoint(Q: np.ndarray, path: Path) -> None:
    """
    Persist a Q-table to disk as a compressed .npz.

    Args:
        Q: 2D NumPy array of shape [n_states, n_actions].
        path: Target file path; parent dirs are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, Q=Q)


def load_checkpoint(path: Path) -> np.ndarray:
    """
    Load a Q-table saved by save_checkpoint.

    Args:
        path: Path to a .npz file containing key 'Q'.

    Returns:
        2D NumPy array (Q-table).
    """
    return np.load(path)["Q"]


def save_manifest(cfg: Any, run_dir: Path) -> None:
    """
    Write a minimal JSON manifest describing a run's key hyperparameters.

    Args:
        cfg: Config-like object with attributes: env_map, episodes, epsilon_schedule,
             alpha, gamma, seed, run_dir, optimistic_init, alpha_schedule, alpha_min.
        run_dir: Run directory (manifest is saved as run_dir/manifest.json).

    Notes:
        • This is complementary to Results/ files and desc.npy; together they make
          runs easy to reproduce and compare.
    """
    manifest = {
        "env_map": cfg.env_map,
        "episodes": cfg.episodes,
        "epsilon_schedule": cfg.epsilon_schedule,
        "alpha": cfg.alpha,
        "gamma": cfg.gamma,
        "seed": cfg.seed,
        "run_dir": str(cfg.run_dir),
        "optimistic_init": float(getattr(cfg, "optimistic_init", 0.0)),
        "alpha_schedule": str(getattr(cfg, "alpha_schedule", "const")),
        "alpha_min": float(getattr(cfg, "alpha_min", 0.1)),
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


class RunningStats:
    """
    Online mean/variance using Welford's algorithm.

    Usage:
        rs = RunningStats()
        for x in stream:
            rs.update(x)
        print(rs.mean, rs.std)

    Attributes:
        n (int): number of observations processed.
        mean (float): running mean of the stream.
        M2 (float): sum of squares of differences from the current mean.
    """

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0  # sum of squares of diffs from mean

    def update(self, x: float) -> None:
        """
        Incorporate a new observation.

        Args:
            x: New sample value.
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        """Return population variance of the processed stream (0.0 if empty)."""
        return self.M2 / self.n if self.n > 0 else 0.0

    @property
    def std(self) -> float:
        """Return population standard deviation of the processed stream."""
        return self.variance ** 0.5
