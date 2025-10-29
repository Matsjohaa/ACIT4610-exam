import numpy as np
from typing import List, Tuple
from .pheromone import PheromoneMatrix
from .heuristic import tight_fit_heuristic

class Ant:
    """An ant that constructs a bin packing solution."""
    
    def __init__(self, items: np.ndarray, capacity: int, alpha: float = 1.0, beta: float = 2.0):
        """
        Initialize an ant.
        
        Args:
            items: Array of item sizes
            capacity: Bin capacity
            alpha: Pheromone influence
            beta: Heuristic influence
        """
        self.items = items
        self.capacity = capacity
        self.n_items = len(items)
        self.alpha = alpha
        self.beta = beta
        
        # Solution: solution[i] = bin index for item i
        self.solution = np.zeros(self.n_items, dtype=int)
        self.n_bins = 0
        self.bin_loads = []
    
    def construct_solution(self, pheromone: PheromoneMatrix, item_order: np.ndarray = None) -> Tuple[np.ndarray, int]:
        """
        Construct a packing solution.
        
        Args:
            pheromone: Pheromone matrix
            item_order: Order to consider items (if None, use given order)
        
        Returns:
            (solution array, number of bins used)
        """
        if item_order is None:
            item_order = np.arange(self.n_items)
        
        # Initialize with one empty bin
        self.bin_loads = [0]
        self.n_bins = 1
        self.solution = np.zeros(self.n_items, dtype=int)
        
        # Pack items one by one
        for item_idx in item_order:
            item_size = self.items[item_idx]
            
            # Get feasible bins (bins where item fits)
            feasible_bins = []
            for b in range(self.n_bins):
                if self.bin_loads[b] + item_size <= self.capacity:
                    feasible_bins.append(b)
            
            # If no feasible bin, open a new one
            if not feasible_bins:
                self.bin_loads.append(0)
                feasible_bins = [self.n_bins]
                self.n_bins += 1
            
            # Compute probabilities for feasible bins
            bin_idx = self._select_bin(item_idx, feasible_bins, item_size, pheromone)
            
            # Assign item to selected bin
            self.solution[item_idx] = bin_idx
            self.bin_loads[bin_idx] += item_size
        
        return self.solution, self.n_bins
    
    def _select_bin(self, item_idx: int, feasible_bins: List[int], 
                    item_size: int, pheromone: PheromoneMatrix) -> int:
        """Select a bin using pheromone and heuristic information."""
        if len(feasible_bins) == 1:
            return feasible_bins[0]
        
        # If alpha=0 and beta=0, use pure random (for testing worst case)
        if self.alpha == 0 and self.beta == 0:
            return np.random.choice(feasible_bins)
        
        # Compute probabilities
        probabilities = []
        
        for b in feasible_bins:
            # Pheromone contribution
            if self.alpha > 0:
                tau = pheromone.get(item_idx, b) ** self.alpha
            else:
                tau = 1.0
            
            # Heuristic contribution
            if self.beta > 0:
                eta = tight_fit_heuristic(item_size, self.bin_loads[b], self.capacity) ** self.beta
            else:
                eta = 1.0
            
            probabilities.append(tau * eta)
        
        probabilities = np.array(probabilities)
        
        # Avoid all-zero probabilities
        if probabilities.sum() == 0:
            probabilities = np.ones(len(feasible_bins))
        
        probabilities /= probabilities.sum()
        
        # Select bin probabilistically
        selected_idx = np.random.choice(len(feasible_bins), p=probabilities)
        return feasible_bins[selected_idx]
    
    def get_fitness(self) -> float:
        """Get solution quality (1 / n_bins for maximization)."""
        return 1.0 / self.n_bins if self.n_bins > 0 else 0