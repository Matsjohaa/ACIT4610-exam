#!/usr/bin/env python3
"""Run a single instance and save the final pheromone matrix.

Usage examples:
  python3 scripts/run_and_save_pheromone.py --instance u500_02
  python3 scripts/run_and_save_pheromone.py --instance u120_00 --preset FAST --seed 2 --out results/pheromone_runs

The script saves:
 - <out_dir>/<timestamp>/<instance>/pheromone_seed<seed>.npy
 - <out_dir>/<timestamp>/<instance>/pheromone_seed<seed>.png

Requires the project root on PYTHONPATH; run from repository root.
"""
import os
import sys
import time
from pathlib import Path
import argparse
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import load_all_instances
from src.algorithm.aco import ACO_BinPacking
from src.algorithm.constants import PRESETS


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def flatten_instances(all_instances) -> dict:
    mapping = {}
    for file_stem, inst_list in all_instances.items():
        for inst in inst_list:
            mapping[inst.name.strip()] = inst
    return mapping


def save_pheromone_and_heatmap(mat: np.ndarray, out_dir: Path, seed: int):
    ensure_dir(out_dir)
    npy_path = out_dir / f'pheromone_seed{seed}.npy'
    png_path = out_dir / f'pheromone_seed{seed}.png'
    # Save raw matrix
    np.save(str(npy_path), mat)
    # Save heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, cmap='viridis', aspect='auto')
    plt.colorbar(label='pheromone')
    plt.title(f'Pheromone heatmap (seed={seed})')
    plt.xlabel('item index')
    plt.ylabel('item index')
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=150)
    plt.close()
    return npy_path, png_path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--instance', required=True, help='Instance name, e.g. u120_00')
    p.add_argument('--preset', default='INTENSIVE', help='Preset name from src.algorithm.constants (default: INTENSIVE)')
    p.add_argument('--seed', type=int, default=1, help='Random seed to use (default: 1)')
    p.add_argument('--out', default='results/pheromone_runs', help='Output base directory')
    p.add_argument('--no-ls', action='store_true', help='Disable local search')
    p.add_argument('--mmas', action='store_true', help='Enable MMAS mode on the solver')
    args = p.parse_args(argv)

    # Prepare output directory
    base_out = Path(args.out)
    ensure_dir(base_out)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_out = base_out / timestamp
    ensure_dir(run_out)

    # Load instances
    all_instances = load_all_instances('data/raw')
    inst_map = flatten_instances(all_instances)

    if args.instance not in inst_map:
        print('Available instances:')
        for k in sorted(inst_map.keys()):
            print(' ', k)
        raise SystemExit(f"Instance '{args.instance}' not found in data/raw")

    inst = inst_map[args.instance]

    # Determine preset
    preset = PRESETS.get(args.preset.upper())
    if preset is None:
        raise SystemExit(f'Unknown preset: {args.preset}')

    # Build solver
    solver = ACO_BinPacking(
        n_ants=int(preset['n_ants']),
        n_iterations=int(preset['n_iterations']),
        alpha=float(preset['alpha']),
        beta=float(preset['beta']),
        rho=float(preset['rho']),
        Q=float(preset['Q']),
    )
    # Optional toggles
    solver.use_local_search = False if args.no_ls else solver.use_local_search
    solver.use_mmas = True if args.mmas else solver.use_mmas

    # Seed RNG
    np.random.seed(int(args.seed))

    print(f"Running instance {inst.name} seed={args.seed} preset={args.preset} ...")
    start = time.time()
    res = solver.solve(inst, logger=None, verbose=False, debug=False)
    elapsed = time.time() - start
    print(f"Done. runtime={elapsed:.3f}s. saving pheromone...")

    # Extract pheromone matrix (ACO returns 'pheromone' key)
    pher = res.get('pheromone')
    if pher is None:
        # try solver attribute
        pher = getattr(getattr(solver, 'pheromone', None), 'pheromone', None)

    if pher is None:
        raise SystemExit('Pheromone matrix not found in solver result or solver. Ensure solver exposes pheromone matrix as `pheromone`.')

    out_dir = run_out / inst.name
    ensure_dir(out_dir)
    npy_path, png_path = save_pheromone_and_heatmap(np.array(pher, dtype=float), out_dir, args.seed)
    print('Saved pheromone matrix to:', npy_path)
    print('Saved heatmap to:', png_path)


if __name__ == '__main__':
    main()
