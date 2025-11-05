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
    
    def construct_solution(self, pheromone: PheromoneMatrix, 
                          item_order: np.ndarray = None) -> Tuple[np.ndarray, int, List[Tuple[int, int]]]:
        """
        Construct a packing solution.
        
        Args:
            pheromone: Pheromone matrix
            item_order: Order to consider items (if None, use given order)
        
        Returns:
            Tuple of (solution array, number of bins used, decisions list)
        """
        if item_order is None:
            item_order = np.arange(self.n_items)
        
        # Initialize with one empty bin
        self.bin_loads = [0]
        self.n_bins = 1
        self.solution = np.zeros(self.n_items, dtype=int)
        decisions: List[Tuple[int, int]] = []
        
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
            
            # Select bin using pheromone and heuristic
            bin_idx = self._select_bin(item_idx, feasible_bins, item_size, pheromone)
            
            # Ensure feasibility: if chosen bin is 'new' or would overflow, open a new bin
            if bin_idx == self.n_bins or (self.bin_loads[bin_idx] + item_size > self.capacity):
                self.bin_loads.append(0)
                bin_idx = self.n_bins
                self.n_bins += 1
            
            # Assign item to selected bin
            self.solution[item_idx] = bin_idx
            self.bin_loads[bin_idx] += item_size
            decisions.append((item_idx, bin_idx))
        
        return self.solution, self.n_bins, decisions
    
    def _select_bin(self, item_idx: int, feasible_bins: List[int], 
                    item_size: int, pheromone: PheromoneMatrix) -> int:
        """
        Select a bin using pheromone and heuristic information.
        
        Returns:
            Selected bin index
        """
        # Trivial case: only one feasible bin
        if len(feasible_bins) == 1:
            return feasible_bins[0]
        
        # Pure random case (for edge cases where alpha=beta=0)
        if self.alpha == 0 and self.beta == 0:
            return np.random.choice(feasible_bins)
        
        # Build candidate set: feasible bins + always 'new bin' option
        candidate_bins: List[int] = feasible_bins.copy()
        new_bin_index = self.n_bins
        candidate_bins.append(new_bin_index)
        
        # Compute probabilities for each candidate
        probabilities = []
        for b in candidate_bins:
            load_before = 0 if b == new_bin_index else self.bin_loads[b]
            
            # Pheromone contribution
            if self.alpha > 0:
                tau_raw = pheromone.get(item_idx, b)
                tau = tau_raw ** self.alpha
            else:
                tau = 1.0
            
            # Heuristic contribution
            if self.beta > 0:
                eta_raw = tight_fit_heuristic(item_size, load_before, self.capacity)
                eta = eta_raw ** self.beta
            else:
                eta = 1.0
            
            probabilities.append(tau * eta)
        
        probabilities = np.array(probabilities)
        
        # Avoid all-zero probabilities
        if probabilities.sum() == 0:
            probabilities = np.ones(len(candidate_bins))
        
        probabilities /= probabilities.sum()
        
        # Select bin probabilistically
        selected_idx = np.random.choice(len(candidate_bins), p=probabilities)
        return candidate_bins[selected_idx]
