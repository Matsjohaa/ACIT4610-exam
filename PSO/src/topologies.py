import numpy as np

def global_best_neighbors(n_particles: int) -> np.ndarray:
    """
    Fully-connected neighborhood matrix for gbest PSO.
    """
    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")
    return np.ones((n_particles, n_particles), dtype=bool)


def ring_lbest_neighbors(n_particles: int, k: int = 2) -> np.ndarray:
    """
    Symmetric ring neighborhood (lbest PSO).

    Each particle i is connected to itself and k/2 neighbors on each side in a ring.
    For example, with k=4, i sees {i-2, i-1, i, i+1, i+2} (mod n).

    Parameters
    ----------
    n_particles : int
        Number of particles in the swarm (must be >= 2).
    k : int, default=2
        Even neighborhood size (must satisfy 2 <= k < n_particles and be even).

    Returns
    -------
    nb : (n_particles, n_particles) bool ndarray
        nb[i, j] == True iff j is in i's local neighborhood.

    Notes
    -----
    - Using k=2 gives immediate left/right neighbors (degree-2 ring).
    - Using k=4 gives two neighbors on each side (degree-4 ring), which is
      commonly used in lbest PSO to balance exploration and information flow.
    """
    if n_particles < 2:
        raise ValueError("n_particles must be >= 2 for a ring topology.")
    if k % 2 != 0:
        raise ValueError("k must be even (neighbors are k/2 on each side).")
    if k < 2:
        raise ValueError("k must be >= 2.")
    if k >= n_particles:
        raise ValueError("k must be less than n_particles to avoid a fully connected graph.")

    nb = np.zeros((n_particles, n_particles), dtype=bool)
    half = k // 2
    for i in range(n_particles):
        nb[i, i] = True
        # wrap-around neighbors
        for d in range(1, half + 1):
            nb[i, (i - d) % n_particles] = True
            nb[i, (i + d) % n_particles] = True
    return nb
