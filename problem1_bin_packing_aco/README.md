# Bin Packing with Ant Colony Optimisation

ACO implementation for the 1-D bin packing benchmarks from the OR-Library. The objective is to use as few bins as possible while respecting the capacity of each bin.

## Data

Benchmark instances are already stored in `data/raw/`. We use the files `binpack1.txt` … `binpack4.txt` (uniform instances with 120–1000 items).

## Quick start

Run commands from the project root (`problem1_bin_packing_aco/`).

```bash
# Quick sanity check (instance u500_02, FAST preset + FFD baseline)
python3 quick_test.py

# Choose a different benchmark instance
python3 quick_test.py --instance u120_00

# Run only ACO (skip baseline heuristics)
python3 quick_test.py --no-baseline

# Compare alternative presets defined in src/algorithm/constants.py
python3 quick_test.py --preset BALANCED
python3 quick_test.py --preset INTENSIVE
```

## Results

- ACO prints summary statistics (bins, gap, runtime, unused capacity) in the terminal.
- Baseline `FFD` is shown for reference so we can judge improvements quickly.

## Presets (see `src/algorithm/constants.py`)

- `QUICK_TEST`: small colony, useful for debugging.
- `FAST`: default preset, uses FFD item order; usually matches FFD quality fast.
- `BALANCED`: disables FFD order to let pheromones learn the packing pattern.
- `INTENSIVE`: large colony and more iterations for deeper exploration.

## Current structure

- `data/raw/`: OR-Library benchmark files (`binpack1.txt` … `binpack4.txt`).
- `results/`: placeholder directories (ignored by Git) for any plots or CSVs you generate.
- `quick_test.py`: simple entry-point that runs ACO (FAST preset) and compares with FFD.
- `src/algorithm/aco.py`: core ACO implementation (pheromone loop, statistics).
- `src/algorithm/components/`: helper classes (`ant.py`, `heuristic.py`, `pheromone.py`).
- `src/algorithm/baseline.py`: greedy baselines (FFD, BFD, WFD).
- `src/algorithm/constants.py`: parameter presets and pheromone bounds.
