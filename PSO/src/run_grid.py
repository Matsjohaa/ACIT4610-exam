from __future__ import annotations
from pathlib import Path
import argparse, sys, os
from math import isfinite

import pandas as pd          # used by boxplot_from_runs (local)
import matplotlib.pyplot as plt
from functions import *
from experiment import run_suite
from config import PSOParams

# ---------- Project-root detection ----------
def find_repo_root(start: Path | None = None) -> Path:
    start = start or Path(__file__).resolve()
    cur = start if start.is_dir() else start.parent
    markers = {"pyproject.toml", "README.md", ".git"}
    for p in [cur, *cur.parents]:
        if any((p / m).exists() for m in markers):
            return p
    return Path(__file__).resolve().parents[2]

REPO_ROOT = find_repo_root()
DEFAULT_OUTDIR = REPO_ROOT / "results"
maybe_src = REPO_ROOT / "src"
if str(maybe_src) not in sys.path:
    sys.path.insert(0, str(maybe_src))

# ---------- Simple boxplot helper  ----------
def boxplot_from_runs(runs_csv: str, outpath: str):
    """Create a compact boxplot of final best fitness per (function, n)."""
    df = pd.read_csv(runs_csv)
    df["combo"] = df["func"] + "_n" + df["n"].astype(str)
    order = sorted(df["combo"].unique())
    data = [df.loc[df["combo"] == c, "best_f"].values for c in order]
    plt.figure()
    plt.boxplot(data, labels=order, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Final best fitness")
    plt.title("PSO final fitness across runs")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="PSO grid runner.")

    # Required-ish
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 10, 30])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--topologies", choices=["gbest", "lbest", "both"], default="both")
    ap.add_argument("--preset", choices=["baseline", "strong"], default="baseline",
                    help="Choose parameter profile (assignment-aligned baseline or stronger).")

    # Optional manual overrides: use None so they only apply if explicitly set
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--swarm", type=int, default=None)
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--c1", type=float, default=None)
    ap.add_argument("--c2", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None, help="vmax fraction of range (e.g., 0.5)")
    ap.add_argument("--ringk", type=int, default=None,
                    help="lbest ring size (even); requires pso_run to read PSOParams.ring_k")
    ap.add_argument("--no-boxplots", dest="no_boxplots", action="store_true")
    return ap.parse_args()

# ---------- Parameter profiles ----------
def make_param_factory(args):
    """
    Returns a callable (n:int, topo:str) -> PSOParams with ALL fields filled.
    Baseline matches the assignment's 'good defaults':
      - swarm: 30 (50 when n=30)
      - iters: 300
      - w=0.7, c1=c2=1.5
      - vmax_frac=0.5
    Strong profile: a popular constriction-like variant.
    """

    def baseline(n: int, topo: str) -> PSOParams:
        swarm = 50 if n == 30 else 30
        return PSOParams(
            swarm_size=swarm,
            iters = 300,
            w=0.7, c1=1.5, c2=1.5,
            vmax_frac=0.5,
            topology=topo,
            seed=None,
            stop_at_threshold=True,
        )

    def strong(n: int, topo: str) -> PSOParams:
        swarm = 50 if n >= 30 else 30
        vmax = 0.3 if n >= 10 else 0.5
        return PSOParams(
            swarm_size=swarm,
            iters=300,
            w=0.729, c1=1.49445, c2=1.49445,
            vmax_frac=vmax,
            topology=topo,
            seed=None,
            stop_at_threshold=True,
        )
    base = baseline if args.preset == "baseline" else strong

    def factory(n: int, topo: str) -> PSOParams:
        p = base(n, topo)
        d = {**p.__dict__}
        if args.iters is not None: d["iters"] = args.iters
        if args.swarm is not None: d["swarm_size"] = args.swarm
        if args.w is not None and isfinite(args.w): d["w"] = args.w
        if args.c1 is not None and isfinite(args.c1): d["c1"] = args.c1
        if args.c2 is not None and isfinite(args.c2): d["c2"] = args.c2
        if args.vmax is not None and isfinite(args.vmax): d["vmax_frac"] = args.vmax
        if args.ringk is not None and topo == "lbest":
            d["ring_k"] = int(args.ringk)
        return PSOParams(**d)

    return factory

def main():
    args = parse_args()
    args.outdir = Path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    param_factory = make_param_factory(args)
    topos = ["gbest", "lbest"] if args.topologies == "both" else [args.topologies]


    for topo in topos:
        runs_csv, summary_csv = run_suite(
            outdir=str(args.outdir),
            dims=tuple(args.dims),
            runs=args.runs,
            topology=topo,
            seed0=args.seed,
            thresholds=SUCCESS_THRESHOLDS,
            param_factory=param_factory,
        )
        if not args.no_boxplots:
            boxplot_from_runs(runs_csv, str(args.outdir / f"boxplot_{topo}.png"))
        print(f"[{topo}] wrote:", runs_csv, summary_csv)

if __name__ == "__main__":
    main()
