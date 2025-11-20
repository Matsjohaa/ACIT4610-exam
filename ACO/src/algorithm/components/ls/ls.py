"""Local search / repair routines for bin packing solutions.

Provides:
- local_repair: multi-pass improvement focusing on moves, merges and swaps.
- local_search_paper: approximation of paper-style replacement LS (open least
    filled bins, attempt structured replacements, reinsert via FFD).

These functions accept a list-of-lists bin structure and return a (possibly)
improved list-of-bins.
"""

from typing import List
import numpy as np


def local_repair(bins: List[List[int]], items: np.ndarray, capacity: int) -> List[List[int]]:
    """
    Improved local repair routine.

    Strategy (multi-pass):
      1. Move items from light bins into the fullest feasible target bins (best-fit by current load).
      2. Attempt to merge light bins into the fullest feasible bin.
      3. Try item swaps between pairs of bins when swapping enables both items to fit.

    The function runs a few passes until no improvement or max_passes reached.
    """
    # Defensive copy
    bins = [list(b) for b in bins]
    n = len(bins)
    bin_loads = [sum(int(items[i]) for i in b) for b in bins]

    max_passes = 8
    pass_no = 0
    while pass_no < max_passes:
        pass_no += 1
        moved = 0
        merged = 0
        swapped = 0

        # Phase 1: move items from light bins to fullest feasible target
        light_bins = sorted(range(len(bins)), key=lambda idx: bin_loads[idx])
        for src in light_bins:
            if not bins[src]:
                continue
            # iterate over a snapshot since we'll modify bins
            for item_idx in list(bins[src]):
                item_size = int(items[item_idx])
                # find feasible targets and prefer fullest (descending load)
                targets = [t for t in range(len(bins)) if t != src and bin_loads[t] + item_size <= capacity]
                if not targets:
                    continue
                targets.sort(key=lambda t: bin_loads[t], reverse=True)
                best = targets[0]
                # move
                try:
                    bins[src].remove(item_idx)
                except ValueError:
                    continue
                bins[best].append(item_idx)
                bin_loads[src] -= item_size
                bin_loads[best] += item_size
                moved += 1

        # Phase 2: merge light bins into fullest feasible target
        # recompute order
        light_bins = sorted(range(len(bins)), key=lambda idx: bin_loads[idx])
        for src in light_bins:
            if not bins[src]:
                continue
            # consider targets sorted by descending load
            targets = sorted([t for t in range(len(bins)) if t != src and bins[t]], key=lambda t: bin_loads[t], reverse=True)
            for t in targets:
                combined = bin_loads[src] + bin_loads[t]
                if combined <= capacity:
                    bins[t].extend(bins[src])
                    bins[src] = []
                    bin_loads[t] = combined
                    bin_loads[src] = 0
                    merged += 1
                    break

        # Phase 3: item-swap moves between bins to enable freeing bins
        # Try swapping a single item from A with single item from B if both fit
        non_empty = [i for i in range(len(bins)) if bins[i]]
        for i_idx in range(len(non_empty)):
            i = non_empty[i_idx]
            for j_idx in range(i_idx + 1, len(non_empty)):
                j = non_empty[j_idx]
                # try swaps
                improved = False
                for a in list(bins[i]):
                    for b in list(bins[j]):
                        a_size = int(items[a])
                        b_size = int(items[b])
                        if (bin_loads[i] - a_size + b_size <= capacity) and (bin_loads[j] - b_size + a_size <= capacity):
                            # perform swap
                            try:
                                bins[i].remove(a)
                                bins[j].remove(b)
                            except ValueError:
                                continue
                            bins[i].append(b)
                            bins[j].append(a)
                            bin_loads[i] = bin_loads[i] - a_size + b_size
                           