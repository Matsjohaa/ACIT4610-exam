import numpy as np
from typing import List
from ..constants import EPSILON

def tight_fit_heuristic(item_size: int, bin_load: int, capacity: int) -> float:
    """
    Weak heuristic - normalized to [0, 1] range.
    
    This gives ACO more room to learn patterns through pheromone trails.
    The heuristic provides a gentle bias toward fuller bins, but doesn't
    dominate the decision-making process.
    
    Args:
        item_size: Size of item to place
        bin_load: Current load in the bin
        capacity: Bin capacity
    
    Returns:
        Heuristic value in [0, 1], where higher = slightly better fit
    """
    remaining = capacity - bin_load
    
    # Item doesn't fit
    if item_size > remaining:
        return 0.0
    
    # Calculate fill ratio after adding item
    # This gives a gentle preference to fuller bins
    fill_ratio = (bin_load + item_size) / capacity
    
    # Return normalized value in (0, 1]
    # Perfect fit gets 1.0, empty bin gets item_size/capacity
    return fill_ratio
