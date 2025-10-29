import numpy as np
from typing import List, Tuple, Dict
import time
from .components.ant import Ant
from .components.pheromone import PheromoneMatrix
from ..data.loader import BinPackingInstance
from .constants import TAU_0

class ACO_BinPacking:
    """Ant Colony Optimization for Bin Packing."""
    
    def __init__(self, 
                 n_ants: int = 20,
                 n_iterations: int = 100,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.1,
                 Q: float = 1.0,
                 use_ffd_order: bool = True):
        """
        Initialize ACO for bin packing.
        
        Args:
            n_ants: Number of ants
            n_iterations: Number of iterations
            alpha: Pheromone importance
            beta: Heuristic importance
            rho: Evaporation rate
            Q: Pheromone deposit factor
            use_ffd_order: Use First-Fit Decreasing order for items
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.use_ffd_order = use_ffd_order
        
        # Statistics
        self.best_solution = None
        self.best_n_bins = float('inf')
        self.convergence_history = []
        self.iteration_best_history = []
    
    def solve(self, instance: BinPackingInstance) -> Dict:
        """
        Solve a bin packing instance.
        
        Args:
            instance: Bin packing instance
        
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        items = instance.items
        capacity = instance.capacity
        n_items = len(items)
        
        # Determine item order (FFD: decreasing size)
        if self.use_ffd_order:
            item_order = np.argsort(items)[::-1]  # Decreasing order
        else:
            item_order = np.arange(n_items)
        
        # Initialize pheromone matrix
        # Max bins upper bound: number of items (worst case)
        max_bins = n_items
        pheromone = PheromoneMatrix(n_items, max_bins, tau_0=TAU_0)
        
        # Statistics
        self.best_solution = None
        self.best_n_bins = float('inf')
        self.convergence_history = []
        self.iteration_best_history = []
        
        # Main ACO loop
        for iteration in range(self.n_iterations):
            # Create ants
            ants = [Ant(items, capacity, self.alpha, self.beta) for _ in range(self.n_ants)]
            
            # Each ant constructs a solution
            iteration_best_bins = float('inf')
            ant_solutions = []
            
            for ant in ants:
                solution, n_bins = ant.construct_solution(pheromone, item_order)
                ant_solutions.append((solution.copy(), n_bins))
                
                # Track best in this iteration
                if n_bins < iteration_best_bins:
                    iteration_best_bins = n_bins
                
                # Track global best
                if n_bins < self.best_n_bins:
                    self.best_n_bins = n_bins
                    self.best_solution = solution.copy()
            
            # Pheromone evaporation
            pheromone.evaporate(self.rho)
            
            # Pheromone deposit (all ants weighted by quality)
            for solution, n_bins in ant_solutions:
                quality = self.Q / n_bins
                pheromone.deposit(solution, quality)
            
            # Record convergence
            self.convergence_history.append(self.best_n_bins)
            self.iteration_best_history.append(iteration_best_bins)
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}: "
                      f"Best={self.best_n_bins}, Iter_best={iteration_best_bins}")
        
        runtime = time.time() - start_time
        
        # Calculate unused capacity
        bin_loads = self._get_bin_loads(self.best_solution, items, capacity)
        total_unused = sum(capacity - load for load in bin_loads)
        
        results = {
            'instance_name': instance.name,
            'n_bins': self.best_n_bins,
            'optimal': instance.optimal,
            'gap': ((self.best_n_bins - instance.optimal) / instance.optimal * 100) if instance.optimal else None,
            'total_unused_capacity': total_unused,
            'runtime': runtime,
            'solution': self.best_solution,
            'bin_loads': bin_loads,
            'convergence': self.convergence_history,
            'iteration_best': self.iteration_best_history
        }
        
        return results
    
    def _get_bin_loads(self, solution: np.ndarray, items: np.ndarray, capacity: int) -> List[int]:
        """Calculate load in each bin."""
        n_bins = solution.max() + 1
        bin_loads = [0] * n_bins
        
        for item_idx, bin_idx in enumerate(solution):
            bin_loads[bin_idx] += items[item_idx]
        
        # Filter out empty bins and sort
        bin_loads = [load for load in bin_loads if load > 0]
        return sorted(bin_loads, reverse=True)