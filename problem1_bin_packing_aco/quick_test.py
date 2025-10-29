#!/usr/bin/env python3
"""Simple one-shot test runner for the bin packing ACO solver."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.loader import load_beasley_format
from src.algorithm.aco import ACO_BinPacking
from src.algorithm.baseline import first_fit_decreasing
from src.algorithm.constants import FAST
from src.logging import RunLogger


def load_instance(name: str):
    """Return a single benchmark instance by name."""
    prefix = name.split('_')[0]
    file_map = {
        'u120': 'binpack1.txt',
        'u250': 'binpack2.txt',
        'u500': 'binpack3.txt',
        'u1000': 'binpack4.txt',
    }
    data_file = file_map.get(prefix)
    if data_file is None:
        raise ValueError("Only the uniform sets u120/u250/u500/u1000 are supported in this quick test.")

    data_path = Path(__file__).parent / "data" / "raw" / data_file
    instances = load_beasley_format(str(data_path))
    for inst in instances:
        if inst.name == name:
            return inst
    raise ValueError(f"Instance {name} not found in {data_file}.")


def main():
    parser = argparse.ArgumentParser(description="Run FAST preset ACO and FFD baseline on one instance")
    parser.add_argument("--instance", default="u500_02", help="Instance name, e.g. u120_00")
    parser.add_argument("--no-baseline", action="store_true", help="Skip the FFD comparison run")
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Optional directory to store iteration-level CSV logs",
    )
    args = parser.parse_args()

    instance = load_instance(args.instance)
    print(f"Instance: {instance.name} | Items: {instance.n_items} | Capacity: {instance.capacity}")

    aco = ACO_BinPacking(**FAST)

    logger = None
    if args.log_dir is not None:
        logger = RunLogger(
            base_dir=args.log_dir,
            filename=f"{instance.name}_aco_log.csv",
            metadata={'runner': 'quick_test'},
        )

    aco_result = aco.solve(
        instance,
        logger=logger,
    )
    print("ACO result")
    print(f"  bins   : {aco_result['n_bins']}")
    if aco_result['gap'] is not None:
        print(f"  gap    : {aco_result['gap']:.2f}%")
    print(f"  runtime: {aco_result['runtime']:.2f}s")

    if not args.no_baseline:
        baseline = first_fit_decreasing(instance)
        print("FFD baseline")
        print(f"  bins   : {baseline['n_bins']}")
        if baseline['gap'] is not None:
            print(f"  gap    : {baseline['gap']:.2f}%")


if __name__ == "__main__":
    main()


