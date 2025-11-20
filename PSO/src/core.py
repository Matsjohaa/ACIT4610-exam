from __future__ import annotations
import numpy as np
from config import PSOParams
from topologies import ring_lbest_neighbors

def _require_params(p: PSOParams):
    missing = [k for k, v in vars(p).items()
               if k in ("swarm_size","iters","w","c1","c2","vmax_frac","topology")
               and v is None]
    if missing:
        raise ValueError(f"PSOParams missing required fields: {missing}")

def pso_run(
    f,
    bounds: tuple[float, float],
    dim: int,
    params: PSOParams,
    rng: np.random.Generator,
    stop_threshold: float | None = 1e-8,  # set None to disable early stop
):
    """Run one PSO trial and return best solution + convergence curve."""
    _require_params(params)

    lo, hi = bounds
    lo_v = np.full(dim, lo, dtype=float)
    hi_v = np.full(dim, hi, dtype=float)
    span = hi_v - lo_v

    if params.vmax_frac is None or params.vmax_frac <= 0:
        raise ValueError("vmax_frac must be > 0.")
    vmax = float(params.vmax_frac) * span

    # Init
    X = rng.uniform(lo_v, hi_v, size=(params.swarm_size, dim))
    V = rng.uniform(-0.2 * span, 0.2 * span, size=(params.swarm_size, dim))  # ~10–20% span
    P = X.copy()
    pbest = np.array([f(x) for x in X], dtype=float)

    topo = (params.topology or "gbest").lower()

    # Neighborhoods
    if topo == "gbest":
        # For gbest we don't really need N; we’ll broadcast the single gbest vector.
        N = None
    elif topo == "lbest":
        k = int(getattr(params, "ring_k", 4))
        N = ring_lbest_neighbors(params.swarm_size, k=k)
    else:
        raise ValueError(f"Unknown topology: {params.topology}")

    # Helper to get local-best indices (lbest)
    def lbest_indices() -> np.ndarray:
        idx = np.empty(params.swarm_size, dtype=int)
        for i in range(params.swarm_size):
            neigh = np.where(N[i])[0]
            idx[i] = neigh[np.argmin(pbest[neigh])]
        return idx

    g_hist = np.empty(params.iters, dtype=float)
    stopped_early = False
    iters_run = 0

    for t in range(params.iters):
        if topo == "gbest":
            g_idx = int(np.argmin(pbest))
            G = P[g_idx]                # shape: (dim,)
        else:
            G = P[lbest_indices()]      # shape: (swarm, dim)

        # Velocity + position update (identical for both topologies)
        r1 = rng.random(size=X.shape)
        r2 = rng.random(size=X.shape)
        # If G is (dim,), NumPy broadcasts to (swarm, dim); if (swarm, dim), it matches directly.
        V = params.w * V + params.c1 * r1 * (P - X) + params.c2 * r2 * (G - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, lo_v, hi_v)

        # Evaluate and update personal bests
        fitness = np.array([f(x) for x in X], dtype=float)
        improved = fitness < pbest
        P[improved] = X[improved]
        pbest[improved] = fitness[improved]

        # Log gbest
        gbest_now = float(np.min(pbest))
        g_hist[t] = gbest_now
        iters_run = t + 1

        # Early stop
        allow_stop = getattr(params, "stop_at_threshold", True)
        if stop_threshold is not None and allow_stop and gbest_now <= stop_threshold:
            stopped_early = True
            break

    # Final stats (+ include initial evaluation cost)
    g_idx = int(np.argmin(pbest))
    evals_used = params.swarm_size * (1 + iters_run)

    return {
        "best_x": P[g_idx].copy(),
        "best_f": float(pbest[g_idx]),
        "gbest_curve": g_hist[:iters_run].copy(),
        "evals_used": int(evals_used),
        "iters_run": int(iters_run),
        "stopped_early": bool(stopped_early),
    }
