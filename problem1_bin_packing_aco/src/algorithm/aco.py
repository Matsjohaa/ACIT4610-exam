"""Ant Colony Optimization (ACO) solver for 1D bin packing.

This module orchestrates the full ACO loop:
- initializes pheromone and per-run settings
- constructs raw solutions with ants (construct phase)
- evaluates and optionally repairs solutions (evaluate phase)
- evaporates and deposits pheromone (update phase)
- tracks global/iteration best, handles stagnation and logging

The main entry point is the ACO_BinPacking class.
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.algorithm.components.ant import Ant
from src.algorithm.components.pheromone import PheromoneMatrix
from src.algorithm import steps as steps
from src.data.loader import BinPackingInstance
from src.algorithm.constants import TAU_0, NO_IMPROVEMENT_LIMIT, DUPLICATE_REPEAT_LIMIT, DUPLICATE_PENALTY_FACTOR
from src.logging import RunLogger
# optional local-search repair routine (best-effort import)
try:
    from src.algorithm.components.ls.ls import local_repair  # type: ignore
except Exception:
    def local_repair(bins, items, capacity):
        return bins


class ACO_BinPacking:
    """Ant Colony Optimization for Bin Packing using grouping-based pheromone."""

    def __init__(self,
                 n_ants: int = 20,
                 n_iterations: int = 100,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 gamma: float = 0.0,
                 rho: float = 0.1,
                 Q: float = 1.0,
                 g: int = 1,
                 exploration_prob: Optional[float] = None):  # frequency of using global-best
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        # Heuristic tight-fit exponent (0 disables blend)
        try:
            self.gamma = max(0.0, float(gamma))
        except Exception:
            self.gamma = 0.0
        self.rho = rho
        self.Q = Q

        # Deposit quality mode: 'falkenauer' (default) or 'inverse_bins'
        self.deposit_quality_mode = 'falkenauer'
        # Per-ant exploration override (None to use module defaults)
        self.exploration_prob = exploration_prob

        # MMAS options
        self.use_mmas = False
        self.pbest = 0.05
        self.g = g

    # Deposit strategy: 'iteration_best_only' (only supported)
        self.deposit_strategy = 'iteration_best_only'

        # Stats
        self.best_solution = None
        self.best_bins: List[List[int]] = []
        self.best_n_bins = float("inf")
        self.convergence_history = []
        self.iteration_best_history = []

        # Stagnation, local-search and duplicate tracking
        self.iterations_since_improvement = 0
        self.ls_trigger = 10
        # Default to clean ACO (no local search). Use runners or CLI flags
        # (e.g. quick_test --no-ls) to enable local search per-run if desired.
        self.use_local_search = False
        self.global_drop_rate = 0.0
        self._seen_structures_counts = {}
        self.seen_structures = set()

        # Whether to apply elitist/global-best deposits (True by default).
        self.elitist = True

        # optionally override pheromone clipping bounds per solver instance
        self.tau_min = None
        self.tau_max = None

        # Optional linear schedules for parameters over iterations.
        # If both *_start and *_end are set, the value will be linearly
        # interpolated each iteration from start->end.
        self.rho_start = None
        self.rho_end = None
        self.alpha_start = None
        self.alpha_end = None
        self.beta_start = None
        self.beta_end = None
        # Optional schedule for gamma (tight-fit heuristic exponent)
        self.gamma_start = None
        self.gamma_end = None

        # Optional schedule for exploration probability
        self.exploration_start = None
        self.exploration_end = None
        # Adaptive exploration controller (optional): increase exploration when
        # pheromone mean collapses near the floor. Clamp with min/max if set.
        self.adaptive_exploration = True
        self.explore_boost_threshold = 0.04  # trigger when mean/tau_max < 4%
        self.explore_boost = 0.15           # min exploration during boost
        self.explore_min = None             # optional hard lower bound
        self.explore_max = None             # optional hard upper bound
        self._last_mean_ratio = None        # set after updates each iteration
        self._ratio_ema = None              # smoothed mean ratio
        self._ratio_ema_alpha = 0.2
        self._adapt_start_frac = 0.2        # enable adaptivity after 20% progress

        # Diversification parameters (used when stagnation smoothing runs)
        self._diversify_iters = 5
        self._diversify_exploration = 0.2
        self._diversify_disable_elitist = True
        self._diversify_counter = 0
        # MMAS safety: prevent extreme collapse by enforcing a minimum
        # tau_min/tau_max ratio when using the internal MMAS bounds.
        # This also serves as a fallback when pbest-based computation is too
        # small for large n (which can drive tau_min ~ 0).
        self.mmas_tau_min_ratio = 0.02  # 2% of tau_max as floor
        self.mmas_tau_min_ratio_base = 0.02
        self.mmas_tau_min_ratio_cap = 0.06  # do not exceed 6%
        self._collapse_streak = 0
        self._collapse_window = 5          # require consecutive low-mean iterations
        # Optional trigger to proactively diversify when mean is near the floor
        # even before the standard NO_IMPROVEMENT limit.
        self._mean_collapse_trigger = 0.025  # if mean/tau_max < 2.5%
        # Logging frequency (iteration-level progress). If >0, only print every log_every iterations.
        self.log_every = 10

    def solve(self,
              instance: BinPackingInstance,
              logger: Optional[RunLogger] = None,
              logger_metadata: Optional[Dict[str, Any]] = None,
              verbose: bool = False,
              debug: bool = False) -> Dict:

        start_time = time.time()
        items = instance.items
        capacity = instance.capacity
        n_items = len(items)

        if logger is not None:
            metadata = {
                "instance_name": instance.name,
                "n_items": n_items,
                "capacity": capacity,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": getattr(self, 'gamma', 0.0),
                "rho": self.rho,
                "Q": self.Q,
                "n_ants": self.n_ants,
                "n_iterations": self.n_iterations,
            }
            if logger_metadata:
                metadata.update(logger_metadata)
            logger.update_metadata(**metadata)

        try:
            import math
            if getattr(self, 'use_mmas', False):
                # approximate tmax as 1/(1-rho) (Stutzle & Hoos approximation)
                tmax = 1.0 / max(1e-12, (1.0 - self.rho))
                # try to compute tmin approximately from pbest, n_items and
                # average branching factor (avg ~ n/2). Use a safe fallback
                # if the formula yields non-positive results.
                avg = max(2.0, float(n_items) / 2.0)
                p = max(1e-9, float(getattr(self, 'pbest', 0.05)))
                numerator = (1.0 / max(1e-12, (1.0 - self.rho))) * max(0.0, (1.0 - float(n_items) * p))
                denom = (avg - 1.0) * float(n_items) * p
                if numerator > 0 and denom > 0:
                    tmin = float(math.sqrt(numerator / denom))
                else:
                    # fallback: set tmin to a safe fraction of tmax (avoid collapse)
                    ratio = float(getattr(self, 'mmas_tau_min_ratio', 0.02))
                    tmin = max(1e-9, tmax * max(1e-4, min(0.25, ratio)))
                tau0 = tmax
            else:
                tmax = self.tau_max if self.tau_max is not None else None
                tmin = self.tau_min if self.tau_min is not None else None
                tau0 = TAU_0
        except Exception:
            # if anything goes wrong, fall back to simple defaults
            tmax = self.tau_max if self.tau_max is not None else None
            tmin = self.tau_min if self.tau_min is not None else None
            tau0 = TAU_0

        pheromone = PheromoneMatrix(n_items, tau_0=tau0, tau_min=tmin, tau_max=tmax)

        # Optional per-run initialization (seed, etc.)
        try:
            steps.initialize_phase(self, items, capacity)
        except Exception:
            pass

        # Helper for linear interpolation in [0,1]
        def _lerp(a: float, b: float, t: float) -> float:
            return a + (b - a) * t

        # Main loop
        for iteration in range(self.n_iterations):
            prev_best = self.best_n_bins
            # progress in [0,1]
            try:
                denom = max(1, self.n_iterations - 1)
                t_prog = float(iteration) / float(denom)
            except Exception:
                t_prog = 0.0

            # Apply alpha/beta schedules if configured
            try:
                if self.alpha_start is not None and self.alpha_end is not None:
                    self.alpha = _lerp(float(self.alpha_start), float(self.alpha_end), t_prog)
            except Exception:
                pass
            try:
                if self.beta_start is not None and self.beta_end is not None:
                    self.beta = _lerp(float(self.beta_start), float(self.beta_end), t_prog)
            except Exception:
                pass
            # Apply gamma schedule if configured
            try:
                if getattr(self, 'gamma_start', None) is not None and getattr(self, 'gamma_end', None) is not None:
                    self.gamma = _lerp(float(self.gamma_start), float(self.gamma_end), t_prog)
                    if self.gamma < 0.0:
                        self.gamma = 0.0
            except Exception:
                pass
            # per-iteration exploration / elitist flags
            # Apply exploration schedule if configured
            cur_exploration = getattr(self, 'exploration_prob', None)
            try:
                if self.exploration_start is not None and self.exploration_end is not None:
                    cur_exploration = _lerp(float(self.exploration_start), float(self.exploration_end), t_prog)
            except Exception:
                pass
            if getattr(self, '_diversify_counter', 0) > 0:
                cur_exploration = max(float(cur_exploration or 0.0), float(getattr(self, '_diversify_exploration', 0.0)))

            # Adaptive exploration based on last iteration's pheromone mean ratio
            try:
                if getattr(self, 'adaptive_exploration', False):
                    last_ratio = getattr(self, '_last_mean_ratio', None)
                    # gate adaptivity until some progress fraction to avoid early flattening
                    adapt_gate = (t_prog >= float(getattr(self, '_adapt_start_frac', 0.2)))
                    if last_ratio is not None and adapt_gate:
                        # smooth the ratio
                        try:
                            if self._ratio_ema is None:
                                self._ratio_ema = float(last_ratio)
                            else:
                                a = float(getattr(self, '_ratio_ema_alpha', 0.2))
                                self._ratio_ema = (1 - a) * float(self._ratio_ema) + a * float(last_ratio)
                        except Exception:
                            self._ratio_ema = float(last_ratio)
                        thr = float(getattr(self, 'explore_boost_threshold', 0.04))
                        if float(self._ratio_ema) < thr:
                            cur_exploration = max(float(cur_exploration or 0.0), float(getattr(self, 'explore_boost', 0.15)))
                    # clamp to explicit bounds if provided
                    if getattr(self, 'explore_min', None) is not None:
                        cur_exploration = max(float(cur_exploration or 0.0), float(self.explore_min))
                    if getattr(self, 'explore_max', None) is not None:
                        cur_exploration = min(float(cur_exploration or 0.0), float(self.explore_max))
            except Exception:
                pass
            # In MMAS mode we disable separate elitist deposits; MMAS alternation already
            # uses either best-of-iteration or best-so-far as sole depositor per iteration.
            active_elitist = (not getattr(self, 'use_mmas', False)) and self.elitist and not (
                getattr(self, '_diversify_counter', 0) > 0 and getattr(self, '_diversify_disable_elitist', True)
            )

            # 1) Construction
            ant_raw = steps.construct_phase(self, pheromone, items, capacity, iteration, cur_exploration, verbose, debug)

            # 2) Evaluate / local-search
            ant_solutions, iteration_best_bins, sorted_solutions = steps.evaluate_phase(self, ant_raw, items, capacity, iteration, verbose, debug)

            # 3) Evaporate + update (deposit + diversification)
            # Compute effective rho (evaporation) using schedule if present
            eff_rho = self.rho
            try:
                if self.rho_start is not None and self.rho_end is not None:
                    eff_rho = _lerp(float(self.rho_start), float(self.rho_end), t_prog)
            except Exception:
                pass
            pheromone.evaporate(eff_rho)
            steps.update_phase(self, pheromone, ant_solutions, items, capacity, iteration, active_elitist, prev_best=prev_best)

            # Proactive anti-collapse: if in MMAS mode and the pheromone mean
            # is very close to the lower bound, gently lift the floor and
            # trigger a brief diversification pulse.
            try:
                # Compute off-diagonal mean ratio for adaptivity
                try:
                    pmean = float(pheromone.mean_offdiag())
                except Exception:
                    pmean = float(pheromone.mean())
                tmax_eff = float(pheromone.tau_max)
                if tmax_eff > 0:
                    ratio = pmean / tmax_eff
                    self._last_mean_ratio = ratio
                else:
                    ratio = 0.0

                # De-escalate tau_min slightly after improvements
                try:
                    if self.best_n_bins < prev_best:
                        base = float(getattr(self, 'mmas_tau_min_ratio_base', 0.02))
                        self.mmas_tau_min_ratio = max(base, float(self.mmas_tau_min_ratio) * 0.9)
                except Exception:
                    pass

                if getattr(self, 'use_mmas', False) and tmax_eff > 0:
                    collapse_thr = float(getattr(self, '_mean_collapse_trigger', 0.025))
                    if ratio < collapse_thr:
                        self._collapse_streak = int(getattr(self, '_collapse_streak', 0)) + 1
                    else:
                        self._collapse_streak = 0

                    # Only escalate if collapse persists and after adaptivity gate
                    adapt_gate = (t_prog >= float(getattr(self, '_adapt_start_frac', 0.2)))
                    if adapt_gate and self._collapse_streak >= int(getattr(self, '_collapse_window', 5)):
                        floor_ratio = float(getattr(self, 'mmas_tau_min_ratio', 0.02))
                        cap = float(getattr(self, 'mmas_tau_min_ratio_cap', 0.06))
                        new_ratio = min(cap, max(float(getattr(self, 'mmas_tau_min_ratio_base', 0.02)), floor_ratio * 1.2))
                        self.mmas_tau_min_ratio = new_ratio
                        new_tmin_target = tmax_eff * max(1e-4, min(0.20, new_ratio))
                        if new_tmin_target > float(pheromone.tau_min):
                            pheromone.tau_min = new_tmin_target
                            import numpy as _np
                            _np.clip(pheromone.pheromone, pheromone.tau_min, pheromone.tau_max, out=pheromone.pheromone)
                            _np.fill_diagonal(pheromone.pheromone, pheromone.tau_0)
                        # Kick a short diversification pulse
                        self._diversify_counter = max(int(getattr(self, '_diversify_counter', 0)), int(getattr(self, '_diversify_iters', 5)))
                        # reset streak
                        self._collapse_streak = 0
            except Exception:
                pass

            if int(getattr(self, 'log_every', 10)) > 0 and (iteration + 1) % int(getattr(self, 'log_every', 10)) == 0:
                pmin, pmax = pheromone.min(), pheromone.max()
                try:
                    pmean = pheromone.mean_offdiag()
                except Exception:
                    pmean = pheromone.mean()
                extras = []
                try:
                    extras.append(f"alpha={float(self.alpha):.3f}")
                except Exception:
                    pass
                try:
                    extras.append(f"beta={float(self.beta):.3f}")
                except Exception:
                    pass
                try:
                    extras.append(f"rho={float(eff_rho):.3f}")
                except Exception:
                    pass
                extra_str = (" | " + ", ".join(extras)) if extras else ""
                # Include off-diagonal mean ratio if available
                ratio_str = ''
                try:
                    if isinstance(pmean, float) and pmax > 0:
                        ratio_str = f" (ratio={pmean/pmax:.3f})"
                except Exception:
                    pass
                print(f"Pheromone min/max/mean: {pmin:.4f}/{pmax:.4f}/{pmean:.4f}{ratio_str}{extra_str}")
            # Decrement diversification counter
            try:
                if getattr(self, '_diversify_counter', 0) > 0:
                    self._diversify_counter -= 1
            except Exception:
                pass

            # Track progress and logging
            self.convergence_history.append(self.best_n_bins)
            self.iteration_best_history.append(iteration_best_bins)

            if logger is not None:
                elapsed_ms = (time.time() - start_time) * 1000
                unused_capacity = self._calc_unused_capacity(self.best_solution, items, capacity)
                logger.log_iteration(
                    iteration=iteration + 1,
                    best_boxes=self.best_n_bins,
                    iteration_best_boxes=iteration_best_bins,
                    unused_capacity=unused_capacity,
                    runtime_ms=elapsed_ms,
                )

            try:
                if int(getattr(self, 'log_every', 10)) > 0 and (iteration + 1) % int(getattr(self, 'log_every', 10)) == 0:
                    print(f"Iter {iteration + 1}/{self.n_iterations}: Best={self.best_n_bins}, IterBest={iteration_best_bins}")
            except Exception:
                pass

        runtime = time.time() - start_time
        bin_loads = self._get_bin_loads(self.best_solution, items, capacity)
        total_unused = sum(capacity - l for l in bin_loads)

        return {
            "instance_name": instance.name,
            "n_bins": self.best_n_bins,
            "optimal": instance.optimal,
            "gap": ((self.best_n_bins - instance.optimal) / instance.optimal * 100)
            if instance.optimal else None,
            "total_unused_capacity": total_unused,
            "runtime": runtime,
            "solution": self.best_solution,
            "bin_loads": bin_loads,
            "convergence": self.convergence_history,
            "iteration_best": self.iteration_best_history,
            # expose final pheromone matrix for external inspection
            "pheromone": pheromone.pheromone.copy(),
        }


    # -----------------------------------------------
    def _get_bin_loads(self, solution: np.ndarray, items: np.ndarray, capacity: int) -> List[int]:
        """Calculate load in each bin."""
        n_bins = solution.max() + 1
        loads = [0] * n_bins
        for i, b in enumerate(solution):
            loads[b] += items[i]
        return [l for l in loads if l > 0]

    def _calc_unused_capacity(self, solution: np.ndarray, items: np.ndarray, capacity: int) -> float:
        loads = self._get_bin_loads(solution, items, capacity)
        return sum(capacity - l for l in loads)