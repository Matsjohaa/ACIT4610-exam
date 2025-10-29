#!/usr/bin/env python3
"""Minimal quick test for the ACO bin packing solver."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.loader import load_beasley_format
from src.algorithm.aco import ACO_BinPacking
from src.algorithm.baseline import first_fit_decreasing
from src.algorithm.constants import FAST


def main():
    parser = argparse.ArgumentParser(description="Run a quick ACO test on one instance")
    parser.add_argument("--instance", default="u500_02", help="Instance name, e.g. u120_00")
    args = parser.parse_args()

    data_path = Path(__file__).parent / "data" / "raw" / "binpack3.txt"
    instances = load_beasley_format(str(data_path))
    instance = next((inst for inst in instances if inst.name == args.instance), None)
    if instance is None:
        raise ValueError(f"Instance {args.instance} not found in {data_path.name}")

    print(f"Instance: {instance.name} | Items: {instance.n_items} | Capacity: {instance.capacity}")

    aco = ACO_BinPacking(**FAST)
    aco_result = aco.solve(instance)
    print(f"ACO bins: {aco_result['n_bins']} (gap {aco_result['gap']:.2f}%)" if aco_result['gap'] is not None else
          f"ACO bins: {aco_result['n_bins']}")

    ffd_result = first_fit_decreasing(instance)
    print(f"FFD bins: {ffd_result['n_bins']} (gap {ffd_result['gap']:.2f}%)" if ffd_result['gap'] is not None else
          f"FFD bins: {ffd_result['n_bins']}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quick test script for ACO Bin Packing algorithm.
Run a single instance and compare ACO vs baseline algorithms.

Usage:
    python3 quick_test.py                                # Test u500_02 with BALANCED preset
    python3 quick_test.py --instance t60_00              # Test specific instance
    python3 quick_test.py --preset FAST                  # Use different preset
    python3 quick_test.py --alpha 1.5 --beta 2.5         # Override specific parameters
    python3 quick_test.py --sweep --preset QUICK_TEST    # Grid-search around a preset
    python3 quick_test.py --visualize                    # Show convergence plot
"""

import sys
import argparse
from pathlib import Path
from itertools import product

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.loader import load_beasley_format
from src.algorithm.aco import ACO_BinPacking
from src.algorithm.baseline import first_fit_decreasing, best_fit_decreasing
from src.algorithm.constants import QUICK_TEST, FAST, BALANCED, INTENSIVE


def print_results(name: str, results: dict):
    """Print results in a nice format."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Instance:        {results['instance_name']}")
    print(f"  Bins used:       {results['n_bins']}")
    print(f"  Optimal:         {results['optimal']}")
    if results['gap'] is not None:
        print(f"  Gap from opt:    {results['gap']:.2f}%")
    print(f"  Unused capacity: {results['total_unused_capacity']}")
    print(f"  Runtime:         {results['runtime']:.4f}s")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Quick test ACO on a bin packing instance")
    parser.add_argument(
        '--instance', 
        type=str, 
        default='u500_02',
        help='Instance name (e.g., u120_00, t60_00)'
    )
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Data file (e.g., binpack1.txt). If not specified, auto-detect from instance name.'
    )
    parser.add_argument(
        '--preset',
        type=str,
        default='BALANCED',
        choices=['QUICK_TEST', 'FAST', 'BALANCED', 'INTENSIVE'],
        help='ACO parameter preset to use'
    )
    # Parameter overrides
    parser.add_argument('--n-ants', type=int, default=None, help='Override number of ants')
    parser.add_argument('--n-iterations', type=int, default=None, help='Override number of iterations')
    parser.add_argument('--alpha', type=float, default=None, help='Override pheromone importance')
    parser.add_argument('--beta', type=float, default=None, help='Override heuristic importance')
    parser.add_argument('--rho', type=float, default=None, help='Override evaporation rate')
    parser.add_argument('--Q', type=float, default=None, help='Override pheromone deposit factor')
    parser.add_argument('--ffd-order', dest='use_ffd_order', action='store_true', help='Force FFD order for items')
    parser.add_argument('--random-order', dest='use_ffd_order', action='store_false', help='Force random item order')
    parser.set_defaults(use_ffd_order=None)
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show convergence visualization'
    )
    parser.add_argument(
        '--no-baseline',
        action='store_true',
        help='Skip baseline algorithms (faster)'
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run grid search over alpha/beta/rho around the chosen preset'
    )
    
    args = parser.parse_args()
    
    # Auto-detect file from instance name
    if args.file is None:
        # Extract prefix (u120, t60, etc.)
        prefix = args.instance.split('_')[0]
        file_map = {
            'u120': 'binpack1.txt',
            'u250': 'binpack2.txt',
            'u500': 'binpack3.txt',
            'u1000': 'binpack4.txt',
            't60': 'binpack5.txt',
            't120': 'binpack6.txt',
            't249': 'binpack7.txt',
            't501': 'binpack8.txt',
        }
        args.file = file_map.get(prefix, 'binpack1.txt')
    
    # Load instance
    data_path = Path(__file__).parent / "data" / "raw" / args.file
    
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        print(f"Run 'python3 scripts/download_data.py' to download data first.")
        return
    
    print(f"📂 Loading instances from {args.file}...")
    instances = load_beasley_format(str(data_path))
    
    # Find the specific instance
    instance = None
    for inst in instances:
        if inst.name == args.instance:
            instance = inst
            break
    
    if instance is None:
        print(f"❌ Error: Instance '{args.instance}' not found in {args.file}")
        print(f"Available instances: {[inst.name for inst in instances[:5]]}...")
        return
    
    print(f"✅ Loaded instance: {instance}")
    
    # Get preset
    presets = {
        'QUICK_TEST': QUICK_TEST,
        'FAST': FAST,
        'BALANCED': BALANCED,
        'INTENSIVE': INTENSIVE,
    }
    params = dict(presets[args.preset])  # copy so we can mutate safely

    # Apply overrides if provided
    if args.n_ants is not None:
        params['n_ants'] = args.n_ants
    if args.n_iterations is not None:
        params['n_iterations'] = args.n_iterations
    if args.alpha is not None:
        params['alpha'] = args.alpha
    if args.beta is not None:
        params['beta'] = args.beta
    if args.rho is not None:
        params['rho'] = args.rho
    if args.Q is not None:
        params['Q'] = args.Q
    if args.use_ffd_order is not None:
        params['use_ffd_order'] = args.use_ffd_order
    
    def run_single(config: dict, label: str = None) -> dict:
        print(f"\n🐜 Running ACO with config: {label or 'custom'}")
        print(f"   Parameters: {config}")
        aco = ACO_BinPacking(**config)
        result = aco.solve(instance)
        print_results("ACO Results", result)
        return result

    if args.sweep:
        base_config = dict(params)
        # Ensure required keys are present
        base_config.setdefault('use_ffd_order', True)
        alpha_values = [1.0, 1.5, 2.0] if args.alpha is None else [args.alpha]
        beta_values = [2.0, 1.5, 2.5] if args.beta is None else [args.beta]
        rho_values = [0.05, 0.1, 0.2] if args.rho is None else [args.rho]

        print("\n🧪 Grid search over alpha, beta, rho")
        print(f"   alpha: {alpha_values}\n   beta: {beta_values}\n   rho: {rho_values}")
        print("\nResults (sorted by bins, then iter_to_best):")
        rows = []
        for alpha_val, beta_val, rho_val in product(alpha_values, beta_values, rho_values):
            sweep_config = dict(base_config)
            sweep_config['alpha'] = alpha_val
            sweep_config['beta'] = beta_val
            sweep_config['rho'] = rho_val
            result = run_single(sweep_config, label=f"alpha={alpha_val}, beta={beta_val}, rho={rho_val}")
            convergence = result['convergence']
            iter_to_best = convergence.index(result['n_bins']) + 1 if result['n_bins'] in convergence else len(convergence)
            rows.append({
                'alpha': alpha_val,
                'beta': beta_val,
                'rho': rho_val,
                'bins': result['n_bins'],
                'gap': result['gap'],
                'iter_to_best': iter_to_best,
                'runtime': result['runtime'],
                'unused': result['total_unused_capacity']
            })

        rows.sort(key=lambda r: (r['bins'], r['iter_to_best'], r['runtime']))
        print("\nSummary (best first):")
        print(f"  {'alpha':>5} {'beta':>5} {'rho':>5} | {'bins':>4} {'gap%':>6} {'iter':>4} {'unused':>7} {'time(s)':>8}")
        print(f"  {'-'*52}")
        for r in rows:
            gap_str = f"{r['gap']:.2f}" if r['gap'] is not None else "NA"
            print(f"  {r['alpha']:5.2f} {r['beta']:5.2f} {r['rho']:5.2f} | {r['bins']:4d} {gap_str:>6} {r['iter_to_best']:4d} {r['unused']:7d} {r['runtime']:8.2f}")

        # Skip baseline in sweep mode unless explicitly requested
        if args.no_baseline:
            return
        else:
            print("\nℹ️ Baselines not run during sweep (use without --sweep for single-config comparison).")
            return

    # Standard single-run behaviour
    aco_results = run_single(params, label=args.preset)
    
    # Run baselines
    if not args.no_baseline:
        print(f"\n📊 Running baseline algorithms for comparison...")
        
        ffd_results = first_fit_decreasing(instance)
        print_results("FFD (First-Fit Decreasing) Baseline", ffd_results)
        
        bfd_results = best_fit_decreasing(instance)
        print_results("BFD (Best-Fit Decreasing) Baseline", bfd_results)
        
        # Summary comparison
        print(f"\n{'='*60}")
        print(f"  SUMMARY COMPARISON")
        print(f"{'='*60}")
        print(f"  Algorithm        Bins    Gap      Runtime")
        print(f"  {'-'*58}")
        print(f"  Optimal          {instance.optimal:4d}    0.00%    -")
        print(f"  ACO              {aco_results['n_bins']:4d}    {aco_results['gap']:5.2f}%   {aco_results['runtime']:7.4f}s")
        print(f"  FFD              {ffd_results['n_bins']:4d}    {ffd_results['gap']:5.2f}%   {ffd_results['runtime']:7.4f}s")
        print(f"  BFD              {bfd_results['n_bins']:4d}    {bfd_results['gap']:5.2f}%   {bfd_results['runtime']:7.4f}s")
        print(f"{'='*60}")
        
        # Winner
        best_bins = min(aco_results['n_bins'], ffd_results['n_bins'], bfd_results['n_bins'])
        winners = []
        if aco_results['n_bins'] == best_bins:
            winners.append("ACO")
        if ffd_results['n_bins'] == best_bins:
            winners.append("FFD")
        if bfd_results['n_bins'] == best_bins:
            winners.append("BFD")
        
        print(f"\n🏆 Best solution: {' = '.join(winners)} with {best_bins} bins")
        
        if best_bins == instance.optimal:
            print(f"✨ Optimal solution found!")
    
    # Visualize
    if args.visualize:
        print(f"\n📈 Generating convergence plot...")
        try:
            from src.visualization.convergence import plot_convergence
            import matplotlib.pyplot as plt

            fig = plot_convergence(
                aco_results['convergence'],
                aco_results['iteration_best'],
                instance.optimal,
                title=f"ACO Convergence on {instance.name}"
            )
            plt.show()
        except ImportError:
            print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"⚠️  Error creating plot: {e}")


if __name__ == "__main__":
    main()


