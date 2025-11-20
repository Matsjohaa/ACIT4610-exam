"""Evaluation (and optional local repair) phase.

Takes raw ant solutions, optionally applies local repair routines, computes
repaired bin counts, updates global/iteration best, and returns enriched
solution tuples for the update (deposit) phase.
"""

from typing import List, Tuple

# Try to use local-search implementations available in components if present.
def _try_local_repair(raw_bins, items, capacity, aco):
    # prefer the components.ls module if available
    try:
        from ..components.ls.ls import ls as ls_module
        if getattr(aco, 'ls_method', 'default') == 'paper' and hasattr(ls_module, 'local_search_paper'):
            return ls_module.local_search_paper(raw_bins, items, capacity, bins_to_open=getattr(aco, 'ls_bins', 4))
        elif hasattr(ls_module, 'local_repair'):
            return ls_module.local_repair(raw_bins, items, capacity)
    except Exception:
        pass

    # try direct import of a local_repair implementation
    try:
        from ..components.ls.ls import local_repair as _lr
        return _lr(raw_bins, items, capacity)
    except Exception:
        pass

    # fallback: return input unchanged
    return raw_bins


def evaluate_phase(aco, ant_raw: List[Tuple], items, capacity, iteration: int, verbose: bool, debug: bool):
    """Attempt local repair on each raw ant solution and evaluate.

    Returns: (ant_solutions, iteration_best_bins, sorted_solutions)
      ant_solutions: list of tuples (solution, raw_n_bins, raw_bins, repaired_bins, repaired_n)
      iteration_best_bins: integer best after repair for this iteration
      sorted_solutions: ant_solutions sorted by repaired_n
    """
    ant_solutions = []
    iteration_best_bins = float('inf')

    for (solution, raw_n_bins, raw_bins, decisions) in ant_raw:
        try:
            if not getattr(aco, 'use_local_search', True):
                repaired_bins = raw_bins
            else:
                repaired_bins = _try_local_repair(raw_bins, items, capacity, aco)

            repaired_n_bins = len(repaired_bins)
        except Exception:
            repaired_bins = raw_bins
            repaired_n_bins = raw_n_bins

        ant_solutions.append((solution, raw_n_bins, raw_bins, repaired_bins, repaired_n_bins))

        # Use repaired counts for iteration-level best
        if repaired_n_bins < iteration_best_bins:
            iteration_best_bins = repaired_n_bins

        # Update global best based on repaired solution
        if repaired_n_bins < aco.best_n_bins:
            aco.best_n_bins = repaired_n_bins
            try:
                aco.best_solution = solution.copy()
            except Exception:
                aco.best_solution = solution
            aco.best_bins = repaired_bins

    sorted_solutions = sorted(ant_solutions, key=lambda x: x[4]) if ant_solutions else []
    return ant_solutions, iteration_best_bins, sorted_solutions

# keep legacy name for compatibility
repair_phase = evaluate_phase
