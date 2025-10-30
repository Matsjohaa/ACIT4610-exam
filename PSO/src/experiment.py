# experiment.py
from __future__ import annotations
import os, csv, json, time
from typing import Callable, Iterable, Mapping
import numpy as np

from config import PSOParams
from functions import FUNCTIONS
from core import pso_run

"""
This file orchestrates runs and persists results in a reproducible way.
"""

def run_suite(
    *,
    outdir: str,
    dims: Iterable[int],
    runs: int,
    topology: str,
    seed0: int,
    thresholds: Mapping[str, float | None],
    param_factory: Callable[[int, str], PSOParams],
):
    """
    Args (all required):
      outdir: output directory for CSVs and curves.
      dims: iterable of dimensions to test (e.g., [2, 10, 30]).
      runs: number of independent runs per (function, n).
      topology: 'gbest' or 'lbest' (used by param_factory; not enforced here).
      seed0: base integer seed to derive per-run RNG seeds deterministically.
      thresholds: dict mapping function name -> success threshold (float) or None.
                  If None, success is not computed (treated as 0 in CSV).
      param_factory: callable (n:int, topology:str) -> PSOParams, producing
                     fully-specified parameters (no None fields).
    Returns:
      (log_csv_path, summary_csv_path)
    """
    # --- Basic checks  ---
    if not isinstance(outdir, str) or not outdir:
        raise ValueError("outdir must be a non-empty string.")
    dims = list(dims)
    if not dims or not all(isinstance(n, int) and n > 0 for n in dims):
        raise ValueError("dims must be a non-empty iterable of positive ints.")
    if not isinstance(runs, int) or runs <= 0:
        raise ValueError("runs must be a positive int.")
    if topology.lower() not in ("gbest", "lbest"):
        raise ValueError("topology must be 'gbest' or 'lbest'.")
    if not isinstance(seed0, int):
        raise ValueError("seed0 must be an int.")
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a dict mapping function name to float|None.")
    # Ensure every function has an entry (explicit None allowed)
    for fname in FUNCTIONS.keys():
        if fname not in thresholds:
            raise ValueError(f"Missing threshold for function '{fname}' in thresholds.")

    os.makedirs(outdir, exist_ok=True)
    curves_dir = os.path.join(outdir, f"curves_{topology}")
    os.makedirs(curves_dir, exist_ok=True)

    log_path = os.path.join(outdir, f"runs_{topology}.csv")
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        # include best_x for compliance (JSON for readability)
        w.writerow(["func", "n", "run", "best_f", "best_x_json", "evals", "success", "time_s"])

        for fname, meta in FUNCTIONS.items():
            f = meta["f"]
            lo, hi = meta["bounds"]
            thr = thresholds.get(fname, None)  # explicit per run_grid

            for n in dims:
                for r in range(runs):
                    p = param_factory(n, topology)
                    # Seed per (func, n, run, topology) deterministically
                    # (Modulo to keep in RNG 32-bit range)
                    run_seed = seed0 + (hash((fname, n, r, topology)) % (2**31 - 1))
                    rng = np.random.default_rng(run_seed)

                    # --- Run PSO ---
                    t0 = time.time()
                    res = pso_run(
                        f=f,
                        bounds=(lo, hi),
                        dim=n,
                        params=p,
                        rng=rng,
                        stop_threshold=(thr if (thr is not None and getattr(p, "stop_at_threshold", True)) else None),
                    )
                    dt = time.time() - t0

                    # --- Persist per-iteration curve ---
                    curve_path = os.path.join(curves_dir, f"{fname}_n{n}_run{r}.npy")
                    np.save(curve_path, res["gbest_curve"])

                    # --- Write result row ---
                    best_f = float(res["best_f"])
                    success = int(best_f <= thr) if thr is not None else 0
                    w.writerow([
                        fname,
                        n,
                        r,
                        best_f,
                        json.dumps([float(v) for v in res["best_x"]]),
                        int(res["evals_used"]),
                        success,
                        float(dt),
                    ])

    agg_path = os.path.join(outdir, f"summary_{topology}.csv")
    _aggregate(log_path, agg_path)
    return log_path, agg_path


def _aggregate(log_csv: str, out_csv: str):
    import pandas as pd
    df = pd.read_csv(log_csv)
    g = df.groupby(["func", "n"], as_index=False)
    summ = g["best_f"].agg(mean="mean", median="median", min="min", max="max", std="std")
    sr = g["success"].mean().rename(columns={"success": "success_rate"})
    out = pd.merge(summ, sr, on=["func", "n"])
    out.to_csv(out_csv, index=False)
