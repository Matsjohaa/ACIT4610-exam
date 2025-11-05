"""Presets and constants for the ACO bin packing solver."""


# ============= Presets =============

# Quick smoke test: small colony, high evaporation, light heuristic weight.
QUICK_TEST = {
    'n_ants': 8,
    'n_iterations': 40,
    'alpha': 1.0,
    'beta': 2.0,  
    'rho': 0.30,    
    'Q': 10.0,      
}

# Default preset: moderate colony, balanced pheromone/heuristic mix.
FAST = {
    'n_ants': 24,
    'n_iterations': 120,
    'alpha': 1.0,
    'beta': 2.0,
    'rho': 0.20,
    'Q': 10.0,
}

# Balanced exploration: larger colony with slower evaporation.
BALANCED = {
    'n_ants': 32,
    'n_iterations': 200,
    'alpha': 1.0,
    'beta': 1.5,    
    'rho': 0.15,
    'Q': 10.0,
}

# Intensive search: deep run with strong pheromone exploitation.
INTENSIVE = {
    'n_ants': 48,
    'n_iterations': 500,   
    'alpha': 1.0,
    'beta': 1.0,    
    'rho': 0.10,    
    'Q': 10.0,
}


# ============= Pheromone Settings =============
# TAU_0: initial pheromone level for every (item, fill-class) edge. Higher values start with
#        stronger bias towards uniform exploration.
# TAU_MIN / TAU_MAX: clipping bounds to prevent pheromone from vanishing or exploding,
#        which keeps probabilities numerically stable and avoids stagnation.
TAU_0 = 0.2
TAU_MIN = 0.01
TAU_MAX = 2.0

# Number of discrete fill classes to describe resulting bin fill levels
# after placing an item. Higher granularity can capture more nuanced
# preferences but may require more iterations to learn.
FILL_CLASSES = 10


# ============= Heuristic Settings =============
# EPSILON: tiny constant to avoid division by zero in heuristics or probability scaling.
EPSILON = 1e-6


# Preset lookup helper for convenience in runners / scripts.
PRESETS = {
    'QUICK_TEST': QUICK_TEST,
    'FAST': FAST,
    'BALANCED': BALANCED,
    'INTENSIVE': INTENSIVE,
}
 