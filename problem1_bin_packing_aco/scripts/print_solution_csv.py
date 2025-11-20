#!/usr/bin/env python3
"""Run ACO for N iterations and print the best packing as a TSV/CSV.

Each column is a bin (Bin 1 .. Bin N). Each row lists one item inside the
corresponding bin. Empty cells are left blank.

Usage examples:
    # From the project folder (problem1_bin_packing_aco/):
    python3 print_solution_csv.py --instance u500_02 --preset QUICK_TEST --iterations 120 --n-ants 24 --no-ls

The script will import the local `src` package (same as `quick_test.py`).
"""

import sys
from pathlib import Path
import argparse

# Resolve project root so this script can be moved to `scripts/` safely.
_candidate = Path(__file__).parent
if not (_candidate / "src").exists():
    _candidate = _candidate.parent
sys.path.insert(0, str(_candidate))

from src.data.loader import load_beasley_format
from src.algorithm.aco import ACO_BinPacking


def load_instance(name: str):
    prefix = name.split('_')[0]
    file_map = {
        'u120': 'binpack1.txt',
        'u250': 'binpack2.txt',
        'u500': 'binpack3.txt',
        'u1000': 'binpack4.txt',
    }
    data_file = file_map.get(prefix)
    if data_file is None:
        raise ValueError("Only the uniform sets u120/u250/u500/u1000 are supported.")

    # Determine project root robustly: if this script lives at the project root
    # then Path(__file__).parent contains `src` and `data`; if it lives in
    # a `scripts/` subfolder, use parent.parent instead. This makes the script
    # resilient to being moved around.
    candidate = Path(__file__).parent
    if not (candidate / "src").exists():
        candidate = candidate.parent

    data_path = candidate / "data" / "raw" / data_file
    instances = load_beasley_format(str(data_path))
    for inst in instances:
        if inst.name == name:
            return inst
    raise ValueError(f"Instance {name} not found in {data_file}.")


def write_bins_csv(bins, items, out_path):
    """Write bins (list of lists of item indices) and items array to a CSV file.

    Each column is one bin (Bin 1 .. Bin N). Rows contain item sizes or empty cells.
    """
    import csv

    # Convert to item sizes for each bin
    columns = []
    for b in bins:
        col = [str(items[i]) for i in b]
        columns.append(col)

    max_rows = max((len(c) for c in columns), default=0)

    headers = [f"Bin {i+1}" for i in range(len(columns))]

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for r in range(max_rows):
            row = []
            for c in columns:
                row.append(c[r] if r < len(c) else "")
            writer.writerow(row)

def compute_bin_loads(bins, items):
    """Return list of total load (sum of item sizes) per bin."""
    loads = []
    for b in bins:
        s = 0
        for idx in b:
            s += int(items[idx])
        loads.append(s)
    return loads


def main():
    parser = argparse.ArgumentParser(description="Print best ACO packing as TSV/CSV")
    from src.algorithm.constants import PRESETS

    parser.add_argument("--instance", default="u500_00", help="Instance name, e.g. u120_00")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default=None, help="Optional preset to load (overrides ACO defaults)")
    parser.add_argument("--iterations", type=int, default=None, help="Number of ACO iterations (overrides preset)")
    parser.add_argument("--n-ants", type=int, default=None, help="Number of ants (overrides preset)")
    parser.add_argument("--no-print-solution", action="store_true", help="Run but don't print the TSV (useful for debugging)")
    # ACO parameter overrides from CLI
    parser.add_argument("--alpha", type=float, default=None, help="Alpha (pheromone importance) (overrides preset)")
    parser.add_argument("--beta", type=float, default=None, help="Beta (heuristic importance) (overrides preset)")
    # When None, we'll resolve gamma from preset or default to 0.0 later
    parser.add_argument("--gamma", type=float, default=None, help="Gamma (tight-fit blend exponent; 0 disables blend)")
    parser.add_argument("--gamma-start", dest="gamma_start", type=float, default=None, help="Start value for gamma schedule (linear)")
    parser.add_argument("--gamma-end", dest="gamma_end", type=float, default=None, help="End value for gamma schedule (linear)")
    parser.add_argument("--rho", type=float, default=None, help="Evaporation rate rho (overrides preset)")
    parser.add_argument("--Q", type=float, default=None, help="Pheromone deposit factor Q (overrides preset)")
    # Optional linear schedules for parameters
    parser.add_argument("--rho-start", dest="rho_start", type=float, default=None, help="Start value for rho schedule (linear)")
    parser.add_argument("--rho-end", dest="rho_end", type=float, default=None, help="End value for rho schedule (linear)")
    parser.add_argument("--alpha-start", dest="alpha_start", type=float, default=None, help="Start value for alpha schedule (linear)")
    parser.add_argument("--alpha-end", dest="alpha_end", type=float, default=None, help="End value for alpha schedule (linear)")
    parser.add_argument("--beta-start", dest="beta_start", type=float, default=None, help="Start value for beta schedule (linear)")
    parser.add_argument("--beta-end", dest="beta_end", type=float, default=None, help="End value for beta schedule (linear)")
    parser.add_argument("--tau-min", type=float, default=None, help="Optional tau_min override (float)")
    parser.add_argument("--tau-max", type=float, default=None, help="Optional tau_max override (float)")
    parser.add_argument("--elitist", dest='elitist', action='store_true', help="Enable elitist/global-best deposits")
    parser.add_argument("--no-elitist", dest='elitist', action='store_false', help="Disable elitist/global-best deposits")
    parser.set_defaults(elitist=True)
    # MMAS toggles
    parser.add_argument("--mmas", dest='mmas', action='store_true', help="Enable MMAS-style tau0=tau_max and compute tau_min from pbest")
    parser.add_argument("--no-mmas", dest='mmas', action='store_false', help="Disable MMAS (use classic ACO tau0)")
    parser.add_argument("--pbest", type=float, default=0.05, help="pbest value to compute tau_min when --mmas is used (default: 0.05)")
    parser.add_argument("--g", type=int, default=5, help="Frequency g: use global-best every g-th iteration (default: 5)")
    parser.set_defaults(mmas=False)
    # allow overriding exploration probability (CLI -> preset -> constant)
    parser.add_argument("--exploration-prob", type=float, default=None, help="Exploration probability override (overrides preset)")
    parser.add_argument("--exploration-start", dest="exploration_start", type=float, default=None, help="Start value for exploration schedule (linear)")
    parser.add_argument("--exploration-end", dest="exploration_end", type=float, default=None, help="End value for exploration schedule (linear)")
    # Adaptive exploration controls
    parser.add_argument("--adaptive-exploration", dest='adaptive_exploration', action='store_true', help="Enable adaptive exploration boost when pheromone mean collapses")
    parser.add_argument("--no-adaptive-exploration", dest='adaptive_exploration', action='store_false', help="Disable adaptive exploration boost")
    parser.set_defaults(adaptive_exploration=True)
    parser.add_argument("--explore-min", type=float, default=None, help="Minimum exploration probability clamp when adaptive exploration is enabled")
    parser.add_argument("--explore-max", type=float, default=None, help="Maximum exploration probability clamp when adaptive exploration is enabled")
    parser.add_argument("--explore-boost-threshold", type=float, default=None, help="Trigger threshold on mean/tau_max ratio for adaptive exploration boost (default 0.04)")
    parser.add_argument("--explore-boost", type=float, default=None, help="Exploration probability to use during adaptive boost (default 0.15)")
    parser.add_argument("--ls-method", choices=["default", "paper"], default="default", help="Local search method to use: 'default' (current) or 'paper' (destroy n least-filled bins).")
    parser.add_argument("--ls-bins", type=int, default=4, help="Number of least-filled bins to open for the paper local search (default: 4)")
    parser.add_argument("--simple-repair", action='store_true', default=False, help="Enable simple/lightweight repair local search (local_repair_light)")
    parser.add_argument("--simple-repair-max-moves", type=int, default=None, help="Max moves for simple repair (default 200)")
    parser.add_argument("--simple-repair-no-merge", dest='simple_repair_no_merge', action='store_true', default=False, help="Disable full-bin merges in simple repair (only individual item moves)")
    parser.add_argument("--simple-repair-final-only", dest='simple_repair_final_only', action='store_true', default=False, help="Only run the simple repair on the final iteration (post-process)")
    parser.add_argument("--simple-repair-every", type=int, default=None, help="Run simple repair every K iterations (integer). If set, overrides final-only.")
    parser.add_argument("--paper-mode", action='store_true', default=False, help="Enable paper-style ACO: single deposit (no elitist), no tau_max clipping (only tau_min), disable adaptive/stagnation extras.")
    parser.add_argument("--paper-tight", action='store_true', default=False, help="In paper-mode, enable tight-fit heuristic (set gamma>0). If --gamma is also provided, that value is used; otherwise defaults to gamma=1.0.")
    parser.add_argument("--no-ls", action='store_true', default=False, help="Disable local search (pure ACO run)")
    # Convenience toggles for this script
    parser.add_argument("--tight-only", action='store_true', default=False, help="Shortcut: set beta=0 to disable size heuristic and rely only on pheromone and tight-fit (gamma)")
    parser.add_argument("--log-convergence", action='store_true', default=False, help="Write per-iteration convergence CSV (best/iterBest/unused/runtime and params)")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory to save convergence CSV (default: <project>/results/logs)")
    parser.add_argument("--log-name", type=str, default=None, help="Optional filename for convergence CSV (default auto-generated)")
    parser.add_argument("--log-diversity", action='store_true', default=False, help="Include extended pheromone diversity metrics in convergence logging (entropy, gini, active fraction, concentration)")
    # Deposit shaping / anti-stagnation extras
    parser.add_argument("--deposit-fullness-power", dest="deposit_fullness_power", type=float, default=None, help="Raise >1.0 to reward perfectly filled bins more than near-full ones during deposit")
    parser.add_argument("--completion-threshold", dest="completion_threshold", type=float, default=None, help="Load fraction threshold to apply completion bonus (e.g., 1.0 for exact full)")
    parser.add_argument("--completion-bonus", dest="completion_bonus", type=float, default=None, help="Multiplicative bonus for bins above completion threshold during deposit")
    parser.add_argument("--quality-unused-k", dest="quality_unused_k", type=float, default=None, help="Penalty factor k for exp(-k * unused/total) applied to solution quality during deposit")
    parser.add_argument("--anti-stagnation-demote", dest="anti_stagnation_demote", type=float, default=None, help="On stagnation, scale down pheromone for pairs in current best bins by this fraction (e.g., 0.1)")
    parser.add_argument("--stagnation-limit", type=int, default=None, help="Custom no-improvement limit to trigger stagnation handling / early stop")
    parser.add_argument("--no-reset-stagnation", dest='no_reset_stagnation', action='store_true', default=False, help="When set, do not reset the internal no-improvement counter when stagnation handling runs (enables early termination if --stagnation-limit is given)")
    parser.add_argument("--reset-mode", choices=["compress","partial_reset","reset_keep_best"], default=None, help="Stagnation reset mode: compress (default), partial_reset (blend with baseline), reset_keep_best (reset then re-deposit best)")
    parser.add_argument("--reset-blend", type=float, default=None, help="Blend factor for partial_reset (0..1, lower = stronger reset toward baseline)")
    parser.add_argument("--reset-best-weight", type=float, default=None, help="Multiplier for re-depositing best solution when using reset_keep_best (0..1)")
    parser.add_argument("--diversify-explore", type=float, default=None, help="Exploration probability to use during stagnation-induced diversification (overrides default)")
    parser.add_argument("--deposit-strategy", choices=["iteration_best_only","global_best_only"], default=None, help="Which solution(s) should receive pheromone deposits: iteration_best_only (default) or global_best_only")
    parser.add_argument(
        "--load-position",
        choices=["above", "below"],
        default="below",
        help="Where to place the total bin loads row: 'above' (under headers) or 'below' (after items). Default: below",
    )
    args = parser.parse_args()

    instance = load_instance(args.instance)
    print(f"Loaded instance {instance.name}: items={instance.n_items}, capacity={instance.capacity}")

    # prepare project results directory and timestamp early so pheromone saving
    # and CSV naming works even if we skip writing the detailed CSV
    project_root = Path(__file__).parent
    if not (project_root / "src").exists():
        project_root = project_root.parent
    results_dir = project_root / "results"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        results_dir = project_root
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve parameters: preset values are used unless CLI overrides provided
    preset_vals = PRESETS.get(args.preset) if args.preset else None

    n_ants = args.n_ants if args.n_ants is not None else (preset_vals['n_ants'] if preset_vals else 20)
    n_iterations = args.iterations if args.iterations is not None else (preset_vals['n_iterations'] if preset_vals else 100)
    alpha = args.alpha if args.alpha is not None else (preset_vals['alpha'] if preset_vals else 1.0)
    # Allow --tight-only to override beta to 0 unless an explicit --beta was given
    if args.beta is not None:
        beta = args.beta
    else:
        beta = (0.0 if bool(args.tight_only) else (preset_vals['beta'] if preset_vals else 2.0))
    gamma = args.gamma if args.gamma is not None else (preset_vals['gamma'] if (preset_vals and 'gamma' in preset_vals) else 0.0)
    rho = args.rho if args.rho is not None else (preset_vals['rho'] if preset_vals else 0.1)
    Q = args.Q if args.Q is not None else (preset_vals['Q'] if preset_vals else 1.0)
    # exploration probability: CLI override -> preset -> None
    exploration = args.exploration_prob if args.exploration_prob is not None else (preset_vals.get('exploration_prob') if preset_vals and 'exploration_prob' in preset_vals else None)

    aco = ACO_BinPacking(
        n_ants=n_ants,
        n_iterations=n_iterations,
        alpha=alpha,
        beta=beta,
        gamma=float(gamma),
        rho=rho,
        Q=Q,
        exploration_prob=exploration,
    )
    # apply optional overrides
    aco.tau_min = args.tau_min
    aco.tau_max = args.tau_max
    aco.elitist = bool(args.elitist)
    # apply optional schedules
    if args.rho_start is not None and args.rho_end is not None:
        aco.rho_start = float(args.rho_start)
        aco.rho_end = float(args.rho_end)
    if args.alpha_start is not None and args.alpha_end is not None:
        aco.alpha_start = float(args.alpha_start)
        aco.alpha_end = float(args.alpha_end)
    if args.beta_start is not None and args.beta_end is not None:
        aco.beta_start = float(args.beta_start)
        aco.beta_end = float(args.beta_end)
    # MMAS-style toggle
    aco.use_mmas = bool(args.mmas)
    if aco.use_mmas:
        aco.pbest = float(args.pbest)
    aco.g = int(args.g)
    aco.ls_method = args.ls_method
    aco.ls_bins = int(args.ls_bins)
    # allow quick toggle for the lightweight/simple repair
    if bool(args.simple_repair):
        aco.ls_method = 'simple'
        # ensure local search is enabled when requesting simple repair
        aco.use_local_search = True
    # configure lightweight repair tuning
    if args.simple_repair_max_moves is not None:
        aco.simple_repair_max_moves = int(args.simple_repair_max_moves)
    if bool(args.simple_repair_no_merge):
        aco.simple_repair_allow_merge = False
    if bool(args.simple_repair_final_only):
        aco.simple_repair_final_only = True
    if args.simple_repair_every is not None:
        aco.simple_repair_every = int(args.simple_repair_every)
    # Paper-mode bundle adjustments
    if bool(args.paper_mode):
        aco.paper_mode = True
        # In paper-mode, allow tight-fit only when requested or explicitly provided
        if args.gamma is not None:
            aco.gamma = float(args.gamma)
        elif bool(args.paper_tight):
            aco.gamma = 1.0  # default tight-fit strength when enabled
        else:
            aco.gamma = 0.0  # default: no tight-fit blend
        # disable elitist extra deposits
        aco.elitist = False
        # disable adaptive exploration & stagnation resets
        aco.adaptive_exploration = False
        aco.reset_on_stagnation = False
        # keep only iteration/global best alternation logic; deposit strategy remains iteration-best by default
        aco.deposit_strategy = 'iteration_best_only'
        # inform solver to construct pheromone matrix without upper clipping
        aco.clip_upper_pheromone = False
        # disable bin load weighting in deposit to match paper's raw co-occurrence reinforcement
        aco.disable_bin_weighting = True
    # allow running pure ACO (disable local search)
    aco.use_local_search = not bool(args.no_ls)
    # stagnation control
    if args.stagnation_limit is not None:
        aco.no_improvement_limit = int(args.stagnation_limit)
    # If the user requests no reset on stagnation, set the flag so the
    # solver will leave `iterations_since_improvement` intact and allow
    # early termination based on `no_improvement_limit`.
    if bool(args.no_reset_stagnation):
        aco.reset_on_stagnation = False
    # Reset mode and diversify exploration overrides
    # Allow quick CLI control of the reset behaviour when stagnation is triggered.
    if hasattr(args, 'reset_mode') and args.reset_mode is not None:
        aco.reset_mode = args.reset_mode
    if hasattr(args, 'reset_blend') and args.reset_blend is not None:
        aco.reset_blend = float(args.reset_blend)
    if hasattr(args, 'reset_best_weight') and args.reset_best_weight is not None:
        aco.reset_best_weight = float(args.reset_best_weight)
    if hasattr(args, 'diversify_explore') and args.diversify_explore is not None:
        aco.diversify_explore = float(args.diversify_explore)
    if hasattr(args, 'deposit_strategy') and args.deposit_strategy is not None:
        aco.deposit_strategy = args.deposit_strategy
    # exploration schedule
    if args.exploration_start is not None and args.exploration_end is not None:
        aco.exploration_start = float(args.exploration_start)
        aco.exploration_end = float(args.exploration_end)
    # adaptive exploration
    aco.adaptive_exploration = bool(args.adaptive_exploration)
    if args.explore_min is not None:
        aco.explore_min = float(args.explore_min)
    if args.explore_max is not None:
        aco.explore_max = float(args.explore_max)
    if args.explore_boost_threshold is not None:
        aco.explore_boost_threshold = float(args.explore_boost_threshold)
    if args.explore_boost is not None:
        aco.explore_boost = float(args.explore_boost)
    # gamma schedule
    if args.gamma_start is not None and args.gamma_end is not None:
        aco.gamma_start = float(args.gamma_start)
        aco.gamma_end = float(args.gamma_end)

    # Deposit shaping / anti-stagnation wiring
    if args.deposit_fullness_power is not None:
        aco.deposit_fullness_power = float(args.deposit_fullness_power)
    if args.completion_threshold is not None:
        aco.deposit_completion_threshold = float(args.completion_threshold)
    if args.completion_bonus is not None:
        aco.deposit_completion_bonus = float(args.completion_bonus)
    if args.quality_unused_k is not None:
        aco.quality_unused_penalty_k = float(args.quality_unused_k)
    if args.anti_stagnation_demote is not None:
        aco.anti_stagnation_demote = float(args.anti_stagnation_demote)
    # Extended diversity logging toggle
    if bool(args.log_diversity):
        aco.log_diversity = True

    # Optional per-iteration CSV logger
    logger = None
    if bool(args.log_convergence):
        from src.logging.run_logger import RunLogger
        # Determine base log directory
        logs_dir = None
        if args.log_dir:
            logs_dir = Path(args.log_dir)
        else:
            default_root = project_root if (project_root / "results").exists() else project_root
            logs_dir = default_root / "results" / "logs"
        # Build a helpful default filename if none provided
        if args.log_name:
            fname = args.log_name
        else:
            # include instance and key params in name
            cfg = f"alpha{alpha}_beta{beta}_gamma{gamma}"
            if args.gamma_start is not None and args.gamma_end is not None:
                cfg = f"alpha{alpha}_beta{beta}_gs{args.gamma_start}_ge{args.gamma_end}"
            fname = f"{instance.name}_{cfg}_{timestamp}.csv"
        logger = RunLogger(base_dir=logs_dir, filename=fname)

    # Run solver (will internally update aco.best_bins)
    if logger is not None:
        result = aco.solve(instance, logger=logger)
        # Flush logger to disk and inform user
        try:
            log_path = logger.flush()
            print(f"Saved convergence CSV to: {log_path}")
        except Exception as e:
            print(f"Warning: could not save convergence CSV: {e}")
    else:
        result = aco.solve(instance)

    # Prefer the stored best_bins (list of item-index lists) if available
    bins = getattr(aco, 'best_bins', None)
    if not bins or len(bins) == 0:
        # Fallback: reconstruct from solution mapping
        sol = result.get('solution')
        if sol is None:
            raise RuntimeError("No solution was found by ACO")
        n_bins = int(max(sol) + 1)
        bins = [[] for _ in range(n_bins)]
        for item_idx, bin_idx in enumerate(sol):
            bins[int(bin_idx)].append(item_idx)

    # Convert item indices to sizes
    items = instance.items

    if not args.no_print_solution:
        # Save CSV to results directory with timestamped filename
        out_fname = f"{instance.name}_aco_packing_{timestamp}.csv"
        out_path = results_dir / out_fname

        # write CSV and include bin loads either above header or below items
        # We'll write a secondary temporary CSV then insert the loads row in the
        # requested position by streaming.
        import csv

        # prepare columns as strings
        columns = []
        for b in bins:
            col = [str(items[i]) for i in b]
            columns.append(col)
        max_rows = max((len(c) for c in columns), default=0)

        headers = [f"Bin {i+1}" for i in range(len(columns))]
        loads = compute_bin_loads(bins, items)

        # write CSV with header, optional loads row, then item rows (or loads after)
        with open(out_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            if args.load_position == 'above':
                writer.writerow([str(l) for l in loads])
            for r in range(max_rows):
                row = [c[r] if r < len(c) else "" for c in columns]
                writer.writerow(row)
            if args.load_position == 'below':
                writer.writerow([str(l) for l in loads])

        print(f"Saved CSV to: {out_path}")

    # Also print summary
    print('\nSummary:')
    print(f"  Best bins: {result['n_bins']}")
    if result.get('gap') is not None:
        print(f"  Gap: {result['gap']:.2f}%")
    print(f"  Runtime: {result['runtime']:.2f}s")

    # Print and save pheromone matrix (if present in result)
    pher = result.get('pheromone')
    if pher is not None:
        try:
            import numpy as _np
            pher_arr = _np.array(pher)
            print(f"\nPheromone matrix (shape={pher_arr.shape}):")
            # Print a compact version: full matrix can be large
            with _np.printoptions(precision=4, suppress=True, threshold=10000):
                print(pher_arr)

            # Save pheromone matrix as CSV too
            pher_fname = f"{instance.name}_pheromone_{timestamp}.csv"
            pher_path = results_dir / pher_fname
            _np.savetxt(pher_path, pher_arr, delimiter=',', fmt='%.6f')
            print(f"Saved pheromone matrix CSV to: {pher_path}")
        except Exception as e:
            print(f"Could not print/save pheromone matrix: {e}")


if __name__ == '__main__':
    main()
