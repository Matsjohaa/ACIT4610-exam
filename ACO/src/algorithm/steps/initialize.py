"""Initialization phase helpers for ACO.

Currently handles RNG seeding for reproducibility. This module is intentionally
lightweight; most setup happens inside the ACO class itself. Serves as a single
place to add per-run initialization (e.g., specialized pheromone seeding).
"""
from typing import Optional
import random


def initialize_phase(aco, items, capacity, seed: Optional[int] = None):
    """Perform any per-run initialization.

    Current behaviour:
    - If `aco` has attribute `random_seed` or a `seed` argument is provided,
      set Python's RNG and optionally numpy's RNG (if available).
    - Return None (placeholder for future initialization values).
    """
    s = seed if seed is not None else getattr(aco, 'random_seed', None)
    if s is not None:
        try:
            random.seed(int(s))
        except Exception:
            pass
        try:
            import numpy as _np
            _np.random.seed(int(s))
        except Exception:
            pass

    # Placeholder: future pheromone initialization helpers could live here.
    return None
