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


def best_fit_decreasing(instance: BinPackingInstance) -> dict:
    """
    Best-Fit Decreasing (BFD) heuristic for bin packing.
    
    Places each item in the bin with the LEAST remaining space that can fit it.
    This tends to pack items more tightly than FFD.
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
        # Find the bin with least remaining space that can fit the item
        best_bin = -1
        min_remaining = float('inf')
        
        for bin_idx in range(len(bins)):
            if bin_loads[bin_idx] + item_size <= capacity:
                remaining = capacity - (bin_loads[bin_idx] + item_size)
                if remaining < min_remaining:
                    min_remaining = remaining
                    best_bin = bin_idx
        
        # Place in best bin or open new one
        if best_bin != -1:
            bins[best_bin].append(idx)
            bin_loads[best_bin] += item_size
            solution[idx] = best_bin
        else:
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


def worst_fit_decreasing(instance: BinPackingInstance) -> dict:
    """
    Worst-Fit Decreasing (WFD) heuristic for bin packing.
    
    Places each item in the bin with the MOST remaining space.
    This tends to balance loads across bins.
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
        # Find the bin with most remaining space that can fit the item
        worst_bin = -1
        max_remaining = -1
        
        for bin_idx in range(len(bins)):
            if bin_loads[bin_idx] + item_size <= capacity:
                remaining = capacity - (bin_loads[bin_idx] + item_size)
                if remaining > max_remaining:
                    max_remaining = remaining
                    worst_bin = bin_idx
        
        # Place in worst bin or open new one
        if worst_bin != -1:
            bins[worst_bin].append(idx)
            bin_loads[worst_bin] += item_size
            solution[idx] = worst_bin
        else:
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