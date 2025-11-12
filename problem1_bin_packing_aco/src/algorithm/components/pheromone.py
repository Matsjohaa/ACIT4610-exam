"""Pheromone matrix for grouping-based ACO in bin packing.

Contains the symmetric τ(i,j) matrix and core operations:
- get: read τ(i,j)
- evaporate: multiplicative decay with clipping to [tau_min, tau_max]
- deposit: distribute a solution quality over item pairs co-occurring in bins,
           optionally weighting by bin fill ratio
"""

import numpy as np
from typing import List, Optional, Iterable, Tuple
try:
    # preferred layout: constants live under core
    from ..constants import TAU_0, TAU_MIN, TAU_MAX
except Exception:
    # backward/older bytecode may reference `..constants`; accept both
    from ..constants import TAU_0, TAU_MIN, TAU_MAX


class PheromoneMatrix:
    """
    Grouping-based pheromone matrix τ(i,j): desirability of placing item i and j
    in the same bin. Matrix is symmetric; diagonal entries are set to tau_0
    (not used).
    """

    def __init__(self,
                 n_items: int,
                 tau_0: float = TAU_0,
                 tau_min: Optional[float] = None,
                 tau_max: Optional[float] = None):
        self.n_items = n_items
        self.tau_0 = float(tau_0)
        self.tau_min = TAU_MIN if tau_min is None else float(tau_min)
        self.tau_max = TAU_MAX if tau_max is None else float(tau_max)

        # symmetric pheromone matrix
        self.pheromone = np.full((n_items, n_items), self.tau_0, dtype=float)
        # keep diagonal at baseline (not used in pair scoring)
        np.fill_diagonal(self.pheromone, self.tau_0)

    def get(self, i: int, j: int) -> float:
        """Return pheromone value τ(i,j)."""
        if i == j:
            return float(self.tau_0)
        return float(self.pheromone[i, j])

    def evaporate(self, rho: float):
        """Evaporate pheromone multiplicatively and clip to bounds."""
        if rho <= 0:
            return
        self.pheromone *= (1.0 - rho)
        np.clip(self.pheromone, self.tau_min, self.tau_max, out=self.pheromone)
        # keep diagonal baseline
        np.fill_diagonal(self.pheromone, self.tau_0)

    def deposit(self,
                structure: Iterable[Iterable[int]],
                quality: float,
                bin_loads: Optional[List[float]] = None,
                capacity: Optional[float] = None):
        """
        Deposit pheromone according to grouping structure.

        - structure: iterable of bins, where each bin is an iterable of item indices.
                     Alternatively accepts a list of (item_idx, bin_idx) decisions
                     (backwards-compatible).
        - quality: base deposit scalar (higher for better solutions)
        - bin_loads: optional list of bin loads aligned with structure
        - capacity: optional, used to weight deposit by fill ratio (bin_load / capacity)

        Deposit strategy:
        - For each bin with m >= 2 items, compute a bin weight w:
            w = 1.0 by default
            if bin_loads and capacity provided: w = bin_load / capacity (clipped to [0,1])
        - Distribute quality * w across the unordered pairs in the bin.
          Each unordered pair (a,b) receives delta = (quality * w) / num_pairs.
        - Update both τ[a,b] and τ[b,a] (matrix kept symmetric).
        """
        # detect legacy "decisions" format: list of (item_idx, bin_idx)
        try:
            # if structure is list of pairs, transform into bins
            if structure and isinstance(next(iter(structure)), tuple) and len(next(iter(structure))) == 2:
                # build bins mapping
                bins_map = {}
                for item_idx, bin_idx in structure:
                    bins_map.setdefault(bin_idx, []).append(item_idx)
                bins = [bins_map[k] for k in sorted(bins_map.keys())]
            else:
                bins = [list(b) for b in structure]
        except StopIteration:
            bins = []

        for idx, bin_items in enumerate(bins):
            m = len(bin_items)
            if m < 2:
                continue
            # compute weight
            w = 1.0
            if bin_loads is not None and capacity:
                try:
                    load = float(bin_loads[idx])
                    if capacity > 0:
                        w = max(0.0, min(1.0, load / float(capacity)))
                except Exception:
                    w = 1.0

            num_pairs = m * (m - 1) / 2.0
            if num_pairs <= 0:
                continue
            delta = (quality * w) / num_pairs

            # add to each unordered pair
            for i_pos in range(m):
                i = bin_items[i_pos]
                for j_pos in range(i_pos + 1, m):
                    j = bin_items[j_pos]
                    # bounds check
                    if 0 <= i < self.n_items and 0 <= j < self.n_items:
                        self.pheromone[i, j] += delta
                        self.pheromone[j, i] += delta

        # clip and enforce diagonal baseline
        np.clip(self.pheromone, self.tau_min, self.tau_max, out=self.pheromone)
        np.fill_diagonal(self.pheromone, self.tau_0)

    def min(self) -> float:
        return float(np.min(self.pheromone))

    def max(self) -> float:
        return float(np.max(self.pheromone))

    def mean(self) -> float:
        return float(np.mean(self.pheromone))

    def mean_offdiag(self) -> float:
        """Return mean pheromone excluding diagonal (baseline entries).

        Diagonal values are fixed at tau_0 and can mask collapse of
        co-occurrence signals. Using the off-diagonal mean gives a
        clearer indicator of exploitation/stagnation.
        """
        try:
            # subtract diagonal sum then divide by off-diagonal count
            total = float(np.sum(self.pheromone))
            diag_sum = float(np.trace(self.pheromone))
            n = self.n_items
            off_count = n * n - n
            if off_count <= 0:
                return float(self.tau_0)
            return (total - diag_sum) / off_count
        except Exception:
            return float(np.mean(self.pheromone))

    def reset(self):
        """Reset pheromone to initial baseline."""
        self.pheromone.fill(self.tau_0)
        np.fill_diagonal(self.pheromone, self.tau_0)