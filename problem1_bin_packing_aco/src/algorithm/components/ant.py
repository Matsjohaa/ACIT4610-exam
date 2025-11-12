"""Ant component: constructs a packing solution guided by pheromone and heuristic.

Implements bin-centric construction procedure:
- build bins one-by-one, repeatedly choosing a feasible item
- selection probability ∝ (avg pheromone to current bin)^alpha * (item size)^beta
- optional exploration, optional local repair

Returns the item-to-bin assignment, bin structure, and the placement decisions.
"""

import numpy as np
from typing import List, Tuple
from pathlib import Path
from .pheromone import PheromoneMatrix
from .heuristic import tight_fit_heuristic


class Ant:
    """
    An ant that constructs a bin packing solution using grouping-based pheromone:
    τ[i][j] = favorability of placing item i and j in the same bin.

    Supports a blended heuristic of the form:
        score(j) ∝ tb(j)^alpha * (item_size^beta) * (tightness(j)^gamma)
    where tightness(j) = (1 - free_space_ratio_after_placement) and gamma >= 0.
    gamma=0 reduces to the previous behaviour (no tight-fit blend).
    """

    def __init__(self, items: np.ndarray, capacity: int, alpha: float = 1.0, beta: float = 2.0, exploration_prob: float = None, gamma: float = 0.0):
        self.items = items
        self.capacity = capacity
        self.n_items = len(items)
        self.alpha = alpha
        self.beta = beta
        # new heuristic exponent for tight-fit factor (gamma)
        self.gamma = max(0.0, float(gamma))
        # per-ant exploration probability (overrides global constant if provided)
        # If None, the ant will fall back to the module-level EXPLORATION_PROB.
        self.exploration_prob = exploration_prob
        self.solution = np.zeros(self.n_items, dtype=int)
        self.n_bins = 0
        self.bin_loads: List[int] = []

    def _compute_score(self, candidate: int, bin_items: List[int], bin_load: int, pheromone: PheromoneMatrix) -> float:
        """Compute the full selection score for a candidate by combining
        pheromone (tb^alpha) with the pure heuristic (size^beta * tightness^gamma).

        The tight-fit heuristic (size, tightness) is computed by
        `tight_fit_heuristic`; this method applies the exponents and pheromone.
        """
        comps = tight_fit_heuristic(candidate=candidate, bin_items=bin_items, items=self.items, bin_load=bin_load, capacity=self.capacity)
        # heuristic returns 0.0 when candidate doesn't fit, otherwise (size, tightness)
        if comps == 0.0:
            return 0.0
        size, tightness = comps

        # apply exponents
        size_term = float(size) ** float(self.beta)
        tight_term = (float(tightness) ** float(self.gamma)) if (self.gamma and float(self.gamma) > 0.0) else 1.0

        # average pheromone to items already in the bin
        if not bin_items:
            tb = float(getattr(pheromone, 'tau_0', 1.0))
        else:
            vals = [pheromone.get(candidate, k) for k in bin_items]
            tb = float(np.mean(vals)) if vals else float(getattr(pheromone, 'tau_0', 1.0))

        return (tb ** float(self.alpha)) * size_term * tight_term

    def construct_solution(self, pheromone: PheromoneMatrix, item_order: np.ndarray = None, current_iter: int = 0, verbose: bool = False, use_local_search: bool = True) -> Tuple[np.ndarray, int, List[Tuple[int, int]]]:
        """
        Construct a packing solution using the paper-style bin-centric
        procedure: fill bins one-by-one. For the current bin, repeatedly
        select the next item from the remaining items that fit using
        probability proportional to tb(j)^alpha * Z(j)^beta where
        tb(j) is the average pheromone between candidate j and items
        already in the bin (or 1.0 if the bin is empty) and Z(j) is the
        item's size (as in the paper).
        """
        # We'll follow the paper: repeatedly fill one bin at a time. Keep a
        # set of remaining items to choose from.
        remaining = set(range(self.n_items))

        bins: List[List[int]] = []
        self.bin_loads = []
        self.solution = np.full(self.n_items, -1)
        decisions: List[Tuple[int, int]] = []

        # prepare optional log file (only when verbose=True) to avoid flooding stdout
        log_f = None
        if verbose:
            log_path = Path(__file__).parents[3] / 'results' / 'tmp_inspect' / 'aco_debug.log'
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_f = open(log_path, 'a', buffering=1)
                write_log = lambda s: log_f.write(s + "\n")
            except Exception:
                # fallback to stdout if file cannot be opened
                log_f = None
                write_log = print
        else:
            write_log = lambda s: None

        from ..constants import EXPLORATION_PROB

        # Build bins until all items are placed
        while remaining:
            # start a new empty bin
            bin_items: List[int] = []
            bin_load = 0

            # repeatedly choose next item for this bin from remaining items that fit
            while True:
                # candidate items that fit
                candidates = [i for i in remaining if int(self.items[i]) + bin_load <= self.capacity]
                if not candidates:
                    break

                # compute full score (pheromone combined with heuristic components)
                scores = []
                for j in candidates:
                    score = self._compute_score(j, bin_items, bin_load, pheromone)
                    scores.append(score)
                scores = np.array(scores, dtype=float)
                if np.all(scores == 0):
                    scores = np.ones_like(scores)

                probs = scores / scores.sum()

                # exploration injection (per-ant override if provided)
                prob = EXPLORATION_PROB if self.exploration_prob is None else float(self.exploration_prob)
                if np.random.rand() < prob:
                    chosen_idx = np.random.choice(len(candidates))
                    write_log(f"[ITER {current_iter}] Exploration: random choose item {candidates[chosen_idx]} for current bin")
                else:
                    chosen_idx = np.random.choice(len(candidates), p=probs)

                chosen_item = candidates[chosen_idx]

                # place chosen item into current bin
                bin_items.append(chosen_item)
                bin_load += int(self.items[chosen_item])
                remaining.remove(chosen_item)
                decisions.append((chosen_item, len(bins)))

                # log concise info
                try:
                    write_log(f"[ITER {current_iter}] Placed item {chosen_item} (size={int(self.items[chosen_item])}) into bin {len(bins)}; bin load {bin_load}/{self.capacity}")
                except Exception:
                    pass

                # continue filling this bin until no candidates

            # finish current bin
            if bin_items:
                bins.append(bin_items)

        # (mapping and bin loads are computed below; local search is handled
        # centrally in the evaluation phase to avoid duplication.)

        # Update stats and assignment after (possible) repair
        self.n_bins = len(bins)
        try:
            self.bin_loads = [int(sum(self.items[i] for i in b)) if b else 0 for b in bins]
        except Exception:
            # fallback: recompute conservatively
            self.bin_loads = [0] * self.n_bins

        # rebuild solution mapping
        self.solution = np.full(self.n_items, -1)
        for b_idx, b in enumerate(bins):
            for item in b:
                try:
                    self.solution[item] = b_idx
                except Exception:
                    pass
        # close log file if opened
        try:
            if log_f is not None:
                log_f.close()
        except Exception:
            pass
        # Return decisions as well for optional debugging/inspection. decisions
        # is a list of tuples (item_idx, chosen_bin) in the order items were
        # placed by this ant.
        return self.solution, self.n_bins, bins, decisions