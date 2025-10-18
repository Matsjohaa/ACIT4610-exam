import numpy as np
from ..constants import TAU_0, TAU_MIN, TAU_MAX

class PheromoneMatrix:
    """Manages pheromone trails for ACO bin packing."""
    
    def __init__(self, n_items: int, max_bins: int, tau_0: float = TAU_0):
        """
        Initialize pheromone matrix.
        
        Args:
            n_items: Number of items to pack
            max_bins: Maximum possible bins (upper bound)
            tau_0: Initial pheromone level
        """
        self.n_items = n_items
        self.max_bins = max_bins
        self.tau_0 = tau_0
        
        # Pheromone[i,b] = desirability of assigning item i to bin b
        self.pheromone = np.ones((n_items, max_bins)) * tau_0
        self.tau_min = TAU_MIN  # Minimum pheromone to prevent stagnation
        self.tau_max = TAU_MAX  # Maximum pheromone
    
    def get(self, item_idx: int, bin_idx: int) -> float:
        """Get pheromone level for (item, bin) pair."""
        return self.pheromone[item_idx, bin_idx]
    
    def evaporate(self, rho: float):
        """
        Evaporate pheromone trails.
        
        Args:
            rho: Evaporation rate (0 < rho < 1)
        """
        self.pheromone *= (1 - rho)
        # Ensure pheromone stays within bounds
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)
    
    def deposit(self, solution: np.ndarray, quality: float):
        """
        Deposit pheromone for a solution.
        
        Args:
            solution: Array where solution[i] = bin assigned to item i
            quality: Quality of solution (typically 1/n_bins)
        """
        for item_idx, bin_idx in enumerate(solution):
            self.pheromone[item_idx, bin_idx] += quality
        
        # Enforce bounds
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)
    
    def reset(self):
        """Reset pheromone to initial values."""
        self.pheromone.fill(self.tau_0)