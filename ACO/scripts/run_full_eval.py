#!/usr/bin/env python3
"""Run batch/full evaluation experiments (default: INTENSIVE preset, no local search).

Saves per-instance results, plots (convergence, load distribution, boxplots)
and a CSV summary under `results/full_evaluation/` by default.
"""
import os
import sys
import time
import csv
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on PYTHONPATH so `src` can be imported when running
# this script directly from the `aco` (project root) folder.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_all_instances
from src.algorithm.aco import ACO_BinPacking
from src.algorithm.baseline import first_fit_decreasing
from src.algorithm.constants import PRESETS
import atexit
from datetime import datetime

# PID / status files for monitoring long runs
RESULTS_ROOT = Path("results/full_evaluation")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def flatten_instances(all_instances) -> dict:
    mapping = {}
    for file_stem, inst_list in all_instances.items():
        for inst in inst_list:
            mapping[inst.name.strip()] = inst
    return mapping


def run_instance(instance, seeds: List[int], preset_name: str = "INTENSIVE", minimal_log: bool = True):
    preset = PRESETS.get(preset_name.upper())
    if preset is None:
        raise ValueError(f"Unknown preset: {preset_name}")

    out_dir = RESULTS_ROOT / instance.name
    ensure_dir(out_dir)
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)

    # Write a simple status file to indicate this instance has started
    status_path = out_dir / f"{instance.name}.status"
    try:
        with open(status_path, 'w') as sf:
            sf.write(f"STARTED: {datetime.utcnow().isoformat()}Z\n")
    except Exception:
        pass

    # Baseline
    baseline_res = first_fit_decreasing(instance)

    runs = []
    for seed in seeds:
        np.random.seed(seed)
        solver = ACO_BinPacking(
            n_ants=int(preset['n_ants']),
            n_iterations=int(preset['n_iterations']),
            alpha=float(preset['alpha']),
            beta=float(preset['beta']),
            rho=float(preset['rho']),
            Q=float(preset['Q'])
        )
        solver.use_local_search = False
        solver.log_every = 0 if minimal_log else 10

        start = time.time()
        res = solver.solve(instance, logger=None, verbose=False, debug=False)
        res['seed'] = seed
        res['run_time_wall'] = time.time() - start
        runs.append(res)

    # Save summary CSV
    csv_path = out_dir / f"summary_{instance.name}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["instance", "seed", "n_bins", "total_unused_capacity", "runtime", "iterations_to_best"]) 
        for r in runs:
            # iterations_to_best: scan convergence for first occurrence of best
            best = r['n_bins']
            try:
                it_best = r['convergence'].index(best) + 1
            except Exception:
                it_best = None
            writer.writerow([instance.name, r['seed'], r['n_bins'], r['total_unused_capacity'], f"{r['runtime']:.4f}", it_best])

    # Save baseline info
    base_csv = out_dir / "baseline.csv"
    with open(base_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["instance", "method", "n_bins", "total_unused_capacity", "runtime"]) 
        writer.writerow([instance.name, "FFD", baseline_res['n_bins'], baseline_res['total_unused_capacity'], f"{baseline_res['runtime']:.4f}"])

    # Mark instance finished in status file
    try:
        with open(status_path, 'a') as sf:
            sf.write(f"FINISHED: {datetime.utcnow().isoformat()}Z\n")
    except Exception:
        pass

    # Plot: convergence (all runs)
    plt.figure(figsize=(6, 4))
    for r in runs:
        plt.plot(range(1, len(r['convergence']) + 1), r['convergence'], color='C0', alpha=0.3)
    # median run
    convs = np.array([r['convergence'] for r in runs])
    if convs.size:
        median_conv = np.median(convs, axis=0)
        plt.plot(range(1, len(median_conv) + 1), median_conv, color='C1', linewidth=2, label='median')
    plt.axhline(y=baseline_res['n_bins'], color='k', linestyle='--', label='FFD')
    plt.xlabel('Iteration')
    plt.ylabel('Best #boxes')
    plt.title(f'Convergence: {instance.name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'convergence.png', dpi=150)
    plt.close()

    # Plot: boxplot of final n_bins
    plt.figure(figsize=(4, 4))
    plt.boxplot([r['n_bins'] for r in runs], labels=['ACO'])
    plt.scatter([1]*len(runs), [r['n_bins'] for r in runs], color='C0', alpha=0.6)
    plt.axhline(y=baseline_res['n_bins'], color='k', linestyle='--', label='FFD')
    plt.title(f'Final #boxes across runs ({instance.name})')
    plt.tight_layout()
    plt.savefig(plots_dir / 'boxplot_final_bins.png', dpi=150)
    plt.close()

    # Plot: load distribution for best overall run
    best_run = min(runs, key=lambda x: x['n_bins'])
    loads = best_run.get('bin_loads', [])
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(loads) + 1), loads)
    plt.axhline(y=instance.capacity, color='k', linestyle=':')
    plt.xlabel('Bin index')
    plt.ylabel('Load')
    plt.title(f'Bin loads (best run) {instance.name} — {best_run["n_bins"]} bins')
    plt.tight_layout()
    plt.savefig(plots_dir / 'load_distribution_best.png', dpi=150)
    plt.close()

    # Optional heatmap: item -> bin assignment (best run)
    solution = best_run.get('solution')
    if solution is not None:
        # create array of bin indices per item in original order
        item_to_bin = np.array(solution)
        # reshape for plotting: sort items by size descending to show packing
        items_sorted_idx = np.argsort(instance.items)[::-1]
        assignment = item_to_bin[items_sorted_idx]
        plt.figure(figsize=(8, 2))
        plt.imshow(assignment[np.newaxis, :], cmap='tab20', aspect='auto')
        plt.yticks([])
        plt.xlabel('Items (sorted desc)')
        plt.title(f'Item->Bin heatmap (best run) {instance.name}')
        plt.colorbar(label='Bin index')
        plt.tight_layout()
        plt.savefig(plots_dir / 'assignment_heatmap.png', dpi=150)
        plt.close()

    return {
        'instance': instance.name,
        'baseline': baseline_res,
        'runs': runs,
        'plots_dir': str(plots_dir),
        'summary_csv': str(csv_path)
    }


def main():
    # Instances to run (match instances listed in the report)
    desired = [
        'u120_00', 'u120_06',
        'u250_00', 'u250_02',
        'u500_00', 'u500_02',
        't249_00', 't501_00',
    ]

    print('Loading instances...')

    # Determine preset: allow overriding via `FAST_EVAL_PRESET` or `PRESET` env vars.
    # Default to `INTENSIVE` since recent parameter sets have been more intensive.
    preset_name = os.getenv('FAST_EVAL_PRESET') or os.getenv('PRESET') or 'INTENSIVE'

    # Configure results folder. Use a canonical `results/full_evaluation`
    # directory but create a timestamped subfolder for each run so outputs
    # are never overwritten.
    global RESULTS_ROOT, PID_FILE
    base_results = Path('results/full_evaluation')
    # Ensure base results dir exists
    ensure_dir(base_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    RESULTS_ROOT = base_results / timestamp
    # PID file placed under `results/` (not per-run folder) for easy monitoring
    PID_FILE = base_results.parent / 'run_full_eval.pid'
    ensure_dir(RESULTS_ROOT)

    # Write PID file for this run
    try:
        ensure_dir(RESULTS_ROOT)
        with open(PID_FILE, 'w') as pf:
            pf.write(str(os.getpid()))
    except Exception:
        pass

    def _cleanup():
        try:
            if PID_FILE.exists():
                PID_FILE.unlink()
        except Exception:
            pass

    atexit.register(_cleanup)
    all_instances = load_all_instances('data/raw')
    inst_map = flatten_instances(all_instances)

    # Allow overriding seeds via environment variable FAST_EVAL_SEEDS (comma-separated)
    env_seeds = os.getenv('FAST_EVAL_SEEDS')
    if env_seeds:
        try:
            seeds = [int(s.strip()) for s in env_seeds.split(',') if s.strip()]
        except Exception:
            seeds = [1]
    else:
        seeds = [1, 2, 3, 4]

    # Allow overriding instances via FAST_INSTANCES env var (comma separated names)
    env_inst = os.getenv('FAST_INSTANCES')
    if env_inst:
        desired = [s.strip() for s in env_inst.split(',') if s.strip()]

    selected = [name for name in desired if name in inst_map]
    missing = [name for name in desired if name not in inst_map]
    if missing:
        print(f"Warning: missing instances (skipping): {missing}")

    ensure_dir(RESULTS_ROOT)
    summary = []
    for name in selected:
        print(f"Running instance {name} with {preset_name} preset; seeds={seeds}...")
        inst = inst_map[name]
        res = run_instance(inst, seeds, preset_name=preset_name, minimal_log=True)
        summary.append(res)

    # Write a master summary CSV
    master_csv = RESULTS_ROOT / 'master_summary.csv'
    with open(master_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['instance', 'seed', 'n_bins', 'total_unused_capacity', 'runtime'])
        for r in summary:
            for run in r['runs']:
                writer.writerow([r['instance'], run['seed'], run['n_bins'], run['total_unused_capacity'], f"{run['runtime']:.4f}"])

    print('Done. Results saved to', RESULTS_ROOT)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Run full evaluation experiments (Intensive preset, no local search).

Saves per-instance results, plots (convergence, load distribution, boxplots)
and a CSV summary under `results/full_evaluation/`.
"""
import os
import sys
import time
import csv
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on PYTHONPATH so `src` can be imported when running
# this script directly from the `aco` (project root) folder.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_all_instances
from src.algorithm.aco import ACO_BinPacking
from src.algorithm.baseline import first_fit_decreasing
from src.algorithm.constants import PRESETS
import atexit
from datetime import datetime

# PID / status files for monitoring long runs
RESULTS_ROOT = Path("results/full_evaluation")

# RESULTS_ROOT and PID_FILE are set in `main()` so the preset can determine
# the output folder (e.g. `results/replicate_intensive`).
PID_FILE = None
RESULTS_ROOT = None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def flatten_instances(all_instances) -> dict:
    mapping = {}
    for file_stem, inst_list in all_instances.items():
        for inst in inst_list:
            mapping[inst.name.strip()] = inst
    return mapping


def run_instance(instance, seeds: List[int], preset_name: str = "INTENSIVE", minimal_log: bool = True):
    preset = PRESETS.get(preset_name.upper())
    if preset is None:
        raise ValueError(f"Unknown preset: {preset_name}")

    out_dir = RESULTS_ROOT / instance.name
    ensure_dir(out_dir)
    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)

    # Write a simple status file to indicate this instance has started
    status_path = out_dir / f"{instance.name}.status"
    try:
        with open(status_path, 'w') as sf:
            sf.write(f"STARTED: {datetime.utcnow().isoformat()}Z\n")
    except Exception:
        pass

    # Baseline
    baseline_res = first_fit_decreasing(instance)

    runs = []
    for seed in seeds:
        np.random.seed(seed)
        solver = ACO_BinPacking(
            n_ants=int(preset['n_ants']),
            n_iterations=int(preset['n_iterations']),
            alpha=float(preset['alpha']),
            beta=float(preset['beta']),
            rho=float(preset['rho']),
            Q=float(preset['Q'])
        )
        solver.use_local_search = False
        solver.log_every = 0 if minimal_log else 10

        start = time.time()
        res = solver.solve(instance, logger=None, verbose=False, debug=False)
        res['seed'] = seed
        res['run_time_wall'] = time.time() - start
        runs.append(res)

    # Save summary CSV
    csv_path = out_dir / f"summary_{instance.name}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["instance", "seed", "n_bins", "total_unused_capacity", "runtime", "iterations_to_best"]) 
        for r in runs:
            # iterations_to_best: scan convergence for first occurrence of best
            best = r['n_bins']
            try:
                it_best = r['convergence'].index(best) + 1
            except Exception:
                it_best = None
            writer.writerow([instance.name, r['seed'], r['n_bins'], r['total_unused_capacity'], f"{r['runtime']:.4f}", it_best])

    # Save baseline info
    base_csv = out_dir / "baseline.csv"
    with open(base_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["instance", "method", "n_bins", "total_unused_capacity", "runtime"]) 
        writer.writerow([instance.name, "FFD", baseline_res['n_bins'], baseline_res['total_unused_capacity'], f"{baseline_res['runtime']:.4f}"])

    # Mark instance finished in status file
    try:
        with open(status_path, 'a') as sf:
            sf.write(f"FINISHED: {datetime.utcnow().isoformat()}Z\n")
    except Exception:
        pass

    # Plot: convergence (all runs)
    plt.figure(figsize=(6, 4))
    for r in runs:
        plt.plot(range(1, len(r['convergence']) + 1), r['convergence'], color='C0', alpha=0.3)
    # median run
    convs = np.array([r['convergence'] for r in runs])
    if convs.size:
        median_conv = np.median(convs, axis=0)
        plt.plot(range(1, len(median_conv) + 1), median_conv, color='C1', linewidth=2, label='median')
    plt.axhline(y=baseline_res['n_bins'], color='k', linestyle='--', label='FFD')
    plt.xlabel('Iteration')
    plt.ylabel('Best #boxes')
    plt.title(f'Convergence: {instance.name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'convergence.png', dpi=150)
    plt.close()

    # Plot: boxplot of final n_bins
    plt.figure(figsize=(4, 4))
    plt.boxplot([r['n_bins'] for r in runs], labels=['ACO'])
    plt.scatter([1]*len(runs), [r['n_bins'] for r in runs], color='C0', alpha=0.6)
    plt.axhline(y=baseline_res['n_bins'], color='k', linestyle='--', label='FFD')
    plt.title(f'Final #boxes across runs ({instance.name})')
    plt.tight_layout()
    plt.savefig(plots_dir / 'boxplot_final_bins.png', dpi=150)
    plt.close()

    # Plot: load distribution for best overall run
    best_run = min(runs, key=lambda x: x['n_bins'])
    loads = best_run.get('bin_loads', [])
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(loads) + 1), loads)
    plt.axhline(y=instance.capacity, color='k', linestyle=':')
    plt.xlabel('Bin index')
    plt.ylabel('Load')
    plt.title(f'Bin loads (best run) {instance.name} — {best_run["n_bins"]} bins')
    plt.tight_layout()
    plt.savefig(plots_dir / 'load_distribution_best.png', dpi=150)
    plt.close()

    # Optional heatmap: item -> bin assignment (best run)
    solution = best_run.get('solution')
    if solution is not None:
        # create array of bin indices per item in original order
        item_to_bin = np.array(solution)
        # reshape for plotting: sort items by size descending to show packing
        items_sorted_idx = np.argsort(instance.items)[::-1]
        assignment = item_to_bin[items_sorted_idx]
        plt.figure(figsize=(8, 2))
        plt.imshow(assignment[np.newaxis, :], cmap='tab20', aspect='auto')
        plt.yticks([])
        plt.xlabel('Items (sorted desc)')
        plt.title(f'Item->Bin heatmap (best run) {instance.name}')
        plt.colorbar(label='Bin index')
        plt.tight_layout()
        plt.savefig(plots_dir / 'assignment_heatmap.png', dpi=150)
        plt.close()

    return {
        'instance': instance.name,
        'baseline': baseline_res,
        'runs': runs,
        'plots_dir': str(plots_dir),
        'summary_csv': str(csv_path)
    }

