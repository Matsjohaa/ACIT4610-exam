import numpy as np
from typing import List
from ..constants import PERFECT_FIT_BONUS, EPSILON

def tight_fit_heuristic(item_size: int, bin_load: int, capacity: int) -> float:
    """
    Heuristic that prefers tight fits.
    
    Higher value = better fit. Prefers bins that will be filled almost completely.
    
    Args:
        item_size: Size of item to place
        bin_load: Current load in the bin
        capacity: Bin capacity
    
    Returns:
        Heuristic value (higher is better)
    """
    remaining = capacity - bin_load
    
    # Item doesn't fit
    if item_size > remaining:
        return 0.0
    
    # Perfect fit gets highest value
    if item_size == remaining:
        return PERFECT_FIT_BONUS
    
    # Otherwise, prefer fuller bins (less wasted space)
    wasted_space = remaining - item_size
    eta = 1.0 / (wasted_space + EPSILON)
    
    return eta


def best_fit_heuristic(item_size: int, bin_loads: List[int], capacity: int) -> np.ndarray:
    """
    Best-fit heuristic for all bins.
    
    Args:
        item_size: Size of item to place
        bin_loads: Current loads in all bins
        capacity: Bin capacity
    
    Returns:
        Array of heuristic values for each bin
    """
    eta = np.zeros(len(bin_loads))
    
    for i, load in enumerate(bin_loads):
        eta[i] = tight_fit_heuristic(item_size, load, capacity)
    
    return eta