"""Greedy baseline algorithms for bin packing.

Provides reference heuristics to compare against ACO. Currently includes
First-Fit Decreasing (FFD).
"""

import numpy as np
from typing import List, Tuple
import time
from ..data.loader import BinPackingInstance

def first_fit_decreasing(instance: BinPackingInstance) -> dict:
    """
    First-Fit Decreasing (FFD) heuristic for bin packing.
    
    A classic greedy baseline algorithm.
    """
    start_time = time.time()
    
    items = instance.items
    capacity = instance.capacity
    n_items = len(items)
    
    # Sort items in decreasing order
    sorted_indices = np.argsort(items)[::-1]
    sorted_items = items[sorted_indices]
    
    # Initialize bins
    bins = []
    bin_loads = []
    solution = np.zeros(n_items, dtype=int)
    
    # Pack each item
    for idx, item_size in zip(sorted_indices, sorted_items):
        # Try to fit in existing bins
        placed = False
        for bin_idx in range(len(bins)):
            if bin_loads[bin_idx] + item_size <= capacity:
                bins[bin_idx].append(idx)
                bin_loads[bin_idx] += item_size
                solution[idx] = bin_idx
                placed = True
                break
        
        # Open new bin if needed
        if not placed:
            bins.append([idx])
            bin_loads.append(item_size)
            solution[idx] = len(bins) - 1
    
    n_bins = len(bins)
    total_unused = sum(capacity - load for load in bin_loads)
    runtime = time.time() - start_time
    
    return {
        'instance_name': instance.name,
        'n_bins': n_bins,
        'optimal': instance.optimal,
        'gap': ((n_bins - instance.optimal) / instance.optimal * 100) if instance.optimal else None,
        'total_unused_capacity': total_unused,
        'runtime': runtime,
        'solution': solution,
        'bin_loads': sorted(bin_loads, reverse=True)
    }
