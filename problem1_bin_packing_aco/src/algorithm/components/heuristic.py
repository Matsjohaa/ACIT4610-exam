import numpy as np
from typing import List


def tight_fit_heuristic(
    candidate: int,
    bin_items: List[int],
    items: np.ndarray,
    bin_load: int,
    capacity: int,
) -> float:
    """Pure heuristic for a tight-fit placement.


    Args:
        candidate: item index to place
        bin_items: items already in the current bin (unused for this pure
            heuristic but kept for API compatibility / future extensions)
        items: array of item sizes
        bin_load: current total load of the bin
        capacity: bin capacity

    Returns:
        A non-negative heuristic value; 0.0 if the item does not fit.
    """

    size = float(items[candidate])
    if bin_load + size > capacity:
        return 0.0

    # Return the fundamental components so the caller can apply exponents
    # (beta, gamma) and combine with pheromone as they see fit.
    # Size (base)
    size_term = size

    # Tightness term (1 - free space ratio after placement)
    free_after = (capacity - bin_load) - size
    free_ratio = max(0.0, free_after / float(capacity))  # clamp to [0,1]
    tightness = max(0.0, 1.0 - free_ratio)

    # Return tuple (size, tightness). Caller applies beta/gamma and pheromone.
    return size_term, tightness
