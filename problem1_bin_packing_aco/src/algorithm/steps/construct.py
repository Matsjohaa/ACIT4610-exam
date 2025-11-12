"""Construction phase: build raw solutions with ants.

Creates `Ant` instances and invokes their bin-centric construction to produce
per-ant raw solutions (solution vector, bin structure, and placement decisions).
No pheromone update occurs here; this step is followed by evaluation/repair.
"""

from typing import List, Tuple
import sys

from ..components.ant import Ant


def construct_phase(aco, pheromone, items, capacity, iteration: int, cur_exploration, verbose: bool, debug: bool) -> List[Tuple]:
    """Construct raw solutions with ants.

    Returns a list of tuples: (solution, raw_n_bins, raw_bins, decisions)
    """
    ant_raw = []
    for i in range(aco.n_ants):
        # pass gamma from solver to ants for blended heuristic
        # pass gamma from solver to ants for blended heuristic
        ant = Ant(items, capacity, aco.alpha, aco.beta, exploration_prob=cur_exploration, gamma=getattr(aco, 'gamma', 0.0))
        solution, raw_n_bins, raw_bins, decisions = ant.construct_solution(
            pheromone,
            current_iter=iteration + 1,
            verbose=verbose,
            use_local_search=getattr(aco, 'use_local_search', False),
        )
        ant_raw.append((solution, raw_n_bins, raw_bins, decisions))

        # preserve the previous debug behaviour: print first ant decisions

        # preserve the previous debug behaviour: print first ant decisions
        if debug and i == 0:
            try:
                print(f"[DEBUG] Iter {iteration+1} - sample ant decisions (item->bin): {decisions}")
            except Exception:
                pass

    return ant_raw
