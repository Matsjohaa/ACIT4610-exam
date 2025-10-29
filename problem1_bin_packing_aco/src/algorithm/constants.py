"""Presets and constants for the ACO bin packing solver."""


# ============= Presets =============

# Quick sanity check: few ants, cheap run, strong heuristic bias.
QUICK_TEST = {
    'n_ants': 10,
    'n_iterations': 50,
    'alpha': 1.0,
    'beta': 2.0,
    'rho': 0.10,
    'Q': 1.0,
    'use_ffd_order': True,
}

# Fast run: moderate ants, modest iterations, balanced pheromone/heuristic mix.
FAST = {
    'n_ants': 20,
    'n_iterations': 75,
    'alpha': 1.2,
    'beta': 1.8,
    'rho': 0.12,
    'Q': 2.0,
    'use_ffd_order': True,
}

# Balanced exploration: more ants and iterations, random item order to force learning.
BALANCED = {
    'n_ants': 28,
    'n_iterations': 140,
    'alpha': 1.3,
    'beta': 1.7,
    'rho': 0.12,
    'Q': 1.5,
    'use_ffd_order': False,
}

# Intensive search: large colony, slow evaporation, pushes pheromone exploitation.
INTENSIVE = {
    'n_ants': 50,
    'n_iterations': 250,
    'alpha': 1.8,
    'beta': 1.4,
    'rho': 0.08,
    'Q': 3.0,
    'use_ffd_order': False,
}


# ============= Pheromone Settings =============
# TAU_0: initial pheromone level for every (item, bin) edge. Higher values start with
#        stronger bias towards uniform exploration.
# TAU_MIN / TAU_MAX: clipping bounds to prevent pheromone from vanishing or exploding,
#        which keeps probabilities numerically stable and avoids stagnation.
TAU_0 = 0.1
TAU_MIN = 0.001
TAU_MAX = 2.0


# ============= Heuristic Settings =============
# EPSILON: tiny constant to avoid division by zero in heuristics or probability scaling.
EPSILON = 1e-6
