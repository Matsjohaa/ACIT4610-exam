"""Update phase: pheromone deposit + stagnation handling.

Implements selection of depositing solutions (strategy/MMAS), quality scoring
using the Falkenauer load-squared metric, duplicate structure penalties,
elitist/global-best deposits, and optional stagnation smoothing/diversification.
Delegates pairwise matrix updates to `PheromoneMatrix.deposit`.
"""

from typing import List, Tuple, Optional
import numpy as np

from ..constants import DUPLICATE_REPEAT_LIMIT, DUPLICATE_PENALTY_FACTOR, NO_IMPROVEMENT_LIMIT


def _deposit_logic(aco, pheromone, ant_solutions: List[Tuple], items, capacity, iteration: int, active_elitist: bool):
    """Internal deposit logic (ported from legacy `deposit.py`)."""
    # sort ants by repaired bin count
    sorted_solutions = sorted(ant_solutions, key=lambda x: x[4]) if ant_solutions else []

    # choose depositors according to strategy (including MMAS alternation)
    depositors = []
    if getattr(aco, 'use_mmas', False) and getattr(aco, 'g', 0) and aco.g > 0:
        if (iteration + 1) % aco.g == 0:
            if aco.best_bins:
                depositors = [(aco.best_solution, aco.best_n_bins, None, aco.best_bins, aco.best_n_bins)]
            else:
                depositors = [sorted_solutions[0]] if sorted_solutions else []
        else:
            depositors = [sorted_solutions[0]] if sorted_solutions else []
    else:
        # Only iteration-best deposits are supported. Always select the
        # best solution from the current iteration (if available).
        depositors = [sorted_solutions[0]] if sorted_solutions else []

    # Log pre/post repair counts for the iteration (if available)
    try:
        if sorted_solutions:
            _, raw_best_bins_count, _, best_repaired_structure, best_repaired_count = sorted_solutions[0]
            log_every = int(getattr(aco, 'log_every', 10))
            if log_every > 0 and ((iteration + 1) % log_every == 0):
                print(f"Iter {iteration+1}/{aco.n_iterations}: RawBins={raw_best_bins_count}, AfterLS={best_repaired_count}, Best={aco.best_n_bins}")
    except Exception:
        pass

    # Deposit pheromone for each selected depositor
    for _, _, _, repaired_structure, rep_n in depositors:
        # deposit pheromone for this depositor
        try:
            # Quality: Falkenauer load-squared metric (fallback to inverse bins on error)
            loads_local = None
            try:
                loads_local = [int(sum(items[item_idx] for item_idx in b)) if b else 0 for b in repaired_structure]
                B = max(1, len([l for l in loads_local if l > 0]))
                f_val = sum(((l / float(capacity)) ** 2) for l in loads_local) / float(B)
                quality = float(aco.Q) * float(f_val)
            except Exception:
                quality = aco.Q / max(1, rep_n)

            # Track repeated structures and optionally penalize
            try:
                struct_key = tuple(sorted(tuple(sorted(b)) for b in repaired_structure))
                aco._seen_structures_counts[struct_key] = aco._seen_structures_counts.get(struct_key, 0) + 1
                count = aco._seen_structures_counts[struct_key]
                try:
                    aco.seen_structures.add(struct_key)
                except Exception:
                    pass
                if count >= DUPLICATE_REPEAT_LIMIT:
                    quality *= DUPLICATE_PENALTY_FACTOR
            except Exception:
                pass

            pheromone.deposit(repaired_structure, quality, bin_loads=loads_local, capacity=capacity)
        except Exception:
            pass

    # Elitist global-best deposit
    # In strict MMAS mode, elitist (extra) deposit is disabled because MMAS already
    # selects either iteration-best or best-so-far as the sole depositor per iteration.
    if (not getattr(aco, 'use_mmas', False)) and active_elitist and getattr(aco, 'best_solution', None) is not None:
        try:
            # Quality for elitist: Falkenauer metric (fallback to inverse bins on error)
            try:
                best_bin_loads = [int(sum(items[i] for i in b)) if b else 0 for b in aco.best_bins]
                B_best = max(1, len([l for l in best_bin_loads if l > 0]))
                f_best = sum(((l / float(capacity)) ** 2) for l in best_bin_loads) / float(B_best)
                quality_elite = float(aco.Q) * float(f_best)
            except Exception:
                quality_elite = aco.Q / max(1, aco.best_n_bins)

            best_bin_loads = [int(sum(items[i] for i in b)) if b else 0 for b in aco.best_bins]
            pheromone.deposit(aco.best_bins, quality_elite, bin_loads=best_bin_loads, capacity=capacity)
        except Exception:
            pass


def stagnation_phase(aco, pheromone, prev_best: Optional[int]):
    """Stagnation handling.

    On long no-improvement stretches, compress pheromone toward a robust
    mid-range and briefly increase exploration to avoid plateaus while
    preserving strict top-only deposits.
    """
    # Update improvement counter
    try:
        if aco.best_n_bins < prev_best:
            aco.iterations_since_improvement = 0
        else:
            aco.iterations_since_improvement += 1
    except Exception:
        # If counters not present, initialize safely
        try:
            aco.iterations_since_improvement = 0
        except Exception:
            pass

    try:
        if getattr(aco, 'iterations_since_improvement', 0) >= NO_IMPROVEMENT_LIMIT:
            try:
                # stagnation handling triggered
                # Compress pheromone toward interquartile midpoint
                M = pheromone.pheromone
                p_low = float(np.percentile(M, 30.0))
                p_high = float(np.percentile(M, 70.0))
                mid = (p_low + p_high) / 2.0
                scale = float(getattr(aco, '_diversify_scale', 0.5))  # 0..1, lower = stronger compression
                M = mid + scale * (M - mid)

                # Add a light jitter to encourage structural diversity
                span = max(1e-12, (pheromone.tau_max - pheromone.tau_min))
                sigma = 0.02 * span
                noise = np.random.normal(loc=0.0, scale=sigma, size=M.shape)
                M = M + noise

                # Clip and keep diagonal baseline
                np.clip(M, pheromone.tau_min, pheromone.tau_max, out=M)
                pheromone.pheromone = M
                np.fill_diagonal(pheromone.pheromone, pheromone.tau_0)

                # Lift the minimum pheromone floor slightly to avoid immediate
                # re-collapse when in MMAS mode or when an extremely low
                # tau_min has been used.
                try:
                    floor_ratio = float(getattr(aco, 'mmas_tau_min_ratio', 0.02))
                    new_tmin = max(float(pheromone.tau_min), float(pheromone.tau_max) * max(1e-4, min(0.25, floor_ratio)))
                    if new_tmin > float(pheromone.tau_min):
                        pheromone.tau_min = new_tmin
                        # Re-clip after raising the floor
                        np.clip(pheromone.pheromone, pheromone.tau_min, pheromone.tau_max, out=pheromone.pheromone)
                        np.fill_diagonal(pheromone.pheromone, pheromone.tau_0)
                except Exception:
                    pass
                # applied compression
            except Exception:
                # Fallback: gentle average to baseline if anything fails
                try:
                    base = getattr(pheromone, 'tau_0', None)
                    if base is None:
                        base = np.mean(pheromone.pheromone)
                    pheromone.pheromone = (pheromone.pheromone + float(base)) / 2.0
                    np.clip(pheromone.pheromone, pheromone.tau_min, pheromone.tau_max, out=pheromone.pheromone)
                    np.fill_diagonal(pheromone.pheromone, pheromone.tau_0)
                except Exception:
                    pass

            # Diversification pulse: increase exploration briefly
            try:
                aco._diversify_counter = max(0, int(getattr(aco, '_diversify_iters', 5)))
            except Exception:
                aco._diversify_counter = 5
            try:
                aco._diversify_exploration = max(float(getattr(aco, '_diversify_exploration', 0.0)), 0.15)
            except Exception:
                aco._diversify_exploration = 0.15
            # triggered diversification

            aco.iterations_since_improvement = 0
    except Exception:
        pass


def update_phase(aco, pheromone, ant_solutions: List[Tuple], items, capacity, iteration: int, active_elitist: bool, prev_best: Optional[int] = None):
    """Combined update phase: deposit + optional stagnation handling."""
    try:
        _deposit_logic(aco, pheromone, ant_solutions, items, capacity, iteration, active_elitist)
    except Exception:
        pass

    try:
        if prev_best is not None:
            stagnation_phase(aco, pheromone, prev_best)
    except Exception:
        pass


# Backwards-compatible names
deposit_phase = update_phase

