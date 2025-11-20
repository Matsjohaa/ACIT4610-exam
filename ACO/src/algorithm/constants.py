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
    'n_ants': 16,
    'n_iterations': 100,
    'alpha': 2.0,
    'beta': 1.0,
    'rho': 0.02,
    'Q': 10.0,
}

# Balanced exploration: larger colony with slower evaporation.
BALANCED = {
    'n_ants': 32,
    'n_iterations': 120,
    'alpha': 2.0,
    'beta': 1.0,    
    'rho': 0.02,
    'Q': 100.0,
}

# Intensive search: deep run with strong pheromone exploitation.
INTENSIVE = {
    'n_ants': 90,
    'n_iterations': 200,   
    'alpha': 2.0,
    'beta': 1.0,    
    'rho': 0.02,    
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


# ============= Stagnation / Duplicate Handling =============
# How many iterations without improvement before triggering stagnation
# handling (compression / diversification).
NO_IMPROVEMENT_LIMIT = 50

# If the same repaired bin-structure has been seen this many times,
# apply a multiplicative penalty to its deposit quality.
DUPLICATE_REPEAT_LIMIT = 4
DUPLICATE_PENALTY_FACTOR = 0.5


# Default per-ant exploration probability (probability to pick random candidate)
# If an Ant is created with `exploration_prob=None`, this module-level default
# will be used.
EXPLORATION_PROB = 0.05


# Preset lookup helper for convenience in runners / scripts.
PRESETS = {
    'QUICK_TEST': QUICK_TEST,
    'FAST': FAST,
    'BALANCED': BALANCED,
    'INTENSIVE': INTENSIVE,
}
 