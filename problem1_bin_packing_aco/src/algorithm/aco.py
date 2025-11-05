import time
from typing import Any, Dict, List, Optional

import numpy as np
from .components.ant import Ant
from .components.pheromone import PheromoneMatrix
from ..data.loader import BinPackingInstance
from .constants import TAU_0
from ..logging import RunLogger

class ACO_BinPacking:
    """Ant Colony Optimization for Bin Packing."""
    
    def __init__(self, 
                 n_ants: int = 20,
                 n_iterations: int = 100,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.1,
                 Q: float = 1.0):
        """
        Initialize ACO for bin packing.
        
        Args:
            n_ants: Number of ants
            n_iterations: Number of iterations
            alpha: Pheromone importance
            beta: Heuristic importance
            rho: Evaporation rate
            Q: Pheromone deposit factor
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # Statistics
        self.best_solution = None
        self.best_n_bins = float('inf')
        self.convergence_history = []
        self.iteration_best_history = []
    
    def solve(
        self,
        instance: BinPackingInstance,
        logger: Optional[RunLogger] = None,
        logger_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
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

        if logger is not None:
            combined_metadata: Dict[str, Any] = {
                'instance_name': instance.name,
                'n_items': n_items,
                'capacity': capacity,
                'alpha': self.alpha,
                'beta': self.beta,
                'rho': self.rho,
                'Q': self.Q,
                'n_ants': self.n_ants,
                'n_iterations': self.n_iterations,
            }
            if logger_metadata:
                combined_metadata.update(logger_metadata)
            logger.update_metadata(**combined_metadata)
        
        # Use original item order (let ACO learn naturally)
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
            
            # Pheromone deposit (only best ant in iteration - as per problem spec)
            best_solution_in_iter = None
            best_n_bins_in_iter = float('inf')
            for solution, n_bins in ant_solutions:
                if n_bins < best_n_bins_in_iter:
                    best_n_bins_in_iter = n_bins
                    best_solution_in_iter = solution
            
            if best_solution_in_iter is not None:
                quality = self.Q / best_n_bins_in_iter
                pheromone.deposit(best_solution_in_iter, quality)
            
            # Record convergence
            self.convergence_history.append(self.best_n_bins)
            self.iteration_best_history.append(iteration_best_bins)

            if logger is not None:
                elapsed_ms = (time.time() - start_time) * 1000
                unused_capacity = None
                if self.best_solution is not None:
                    current_loads = self._get_bin_loads(self.best_solution, items, capacity)
                    unused_capacity = sum(capacity - load for load in current_loads)

                logger.log_iteration(
                    iteration=iteration + 1,
                    best_boxes=self.best_n_bins,
                    iteration_best_boxes=iteration_best_bins,
                    unused_capacity=unused_capacity,
                    runtime_ms=elapsed_ms,
                )
            
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

        if logger is not None:
            logger.flush()
        
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
