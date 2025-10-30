from dataclasses import dataclass
from typing import Optional

"""
Dataclass definition for PSO hyperparameters.

All fields are optional (`None`) so that this file acts only as an override layer.
The desired parameter values and experiment defaults (e.g., swarm size,
iteration count, inertia weight, etc.) are defined in `run_grid.py`.

"""
@dataclass(frozen=True)
class PSOParams:
    swarm_size: Optional[int] = None
    iters: Optional[int] = None
    w: Optional[float] = None
    c1: Optional[float] = None
    c2: Optional[float] = None
    vmax_frac: Optional[float] = None
    topology: Optional[str] = None
    seed: Optional[int] = None
    stop_at_threshold: Optional[bool] = None