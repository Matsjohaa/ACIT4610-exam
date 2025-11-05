import numpy as np
from ..constants import TAU_0, TAU_MIN, TAU_MAX

class PheromoneMatrix:
    """Pheromone over (item, bin_index) pairs."""

    def __init__(self, n_items: int, max_bins: int, tau_0: float = TAU_0):
        self.n_items = n_items
        self.max_bins = max_bins
        self.tau_0 = tau_0

        # pheromone[i, b] = desirability of assigning item i to bin index b
        self.pheromone = np.ones((n_items, max_bins)) * tau_0
        self.tau_min = TAU_MIN
        self.tau_max = TAU_MAX

    def get(self, item_idx: int, bin_idx: int) -> float:
        return self.pheromone[item_idx, bin_idx]

    def evaporate(self, rho: float):
        self.pheromone *= (1 - rho)
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)

    def deposit(self, decisions, quality: float):
        """
        decisions: list of (item_idx, bin_idx) used in the selected solution
        quality: pheromone deposit factor scaled by solution quality
        """
        for item_idx, bin_idx in decisions:
            if 0 <= bin_idx < self.pheromone.shape[1]:
                self.pheromone[item_idx, bin_idx] += quality
        self.pheromone = np.clip(self.pheromone, self.tau_min, self.tau_max)

    def reset(self):
        self.pheromone.fill(self.tau_0)