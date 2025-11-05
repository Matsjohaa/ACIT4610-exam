# Bin Packing with Ant Colony Optimization

ACO implementation for the 1-D bin packing problem using benchmarks from the OR-Library. The goal is to pack items into the minimum number of bins without exceeding bin capacity.

## Algorithm Overview

**Ant Colony Optimization (ACO)** is a metaheuristic inspired by how ants find shortest paths using pheromone trails.

### How it works:

1. **Initialization**: Start with uniform pheromone trails on all (item, bin) assignments.

2. **Construction Phase** (each iteration):

   - Each ant builds a complete packing solution
   - For each item, the ant probabilistically selects a bin based on:
     - **Pheromone (τ)**: What worked well in past iterations
     - **Heuristic (η)**: Tight-fit preference (fuller bins = better)
   - Selection probability: `P(bin) ∝ (τ^α) × (η^β)`
   - Items are processed in random order (different per ant)

3. **Pheromone Update**:

   - **Evaporation**: All pheromone trails decay by factor `(1 - ρ)`
   - **Deposit**: Only the best ant from this iteration reinforces its (item, bin) decisions
   - Deposit amount: `Q / n_bins` (better solutions deposit more)

4. **Convergence**: Over iterations, strong pheromone trails emerge on good assignments, guiding future ants toward high-quality packings.

### Key Parameters:

- `α` (alpha): Pheromone importance (how much we trust past experience)
- `β` (beta): Heuristic importance (how much we follow greedy tight-fit)
- `ρ` (rho): Evaporation rate (higher = faster forgetting)
- `Q`: Pheromone deposit scaling factor
- `n_ants`: Colony size
- `n_iterations`: Number of search iterations

## Data

Benchmark instances from OR-Library are in `data/raw/`:

- `binpack1.txt`: 120-item uniform instances (u120_00 ... u120_19)
- `binpack2.txt`: 250-item uniform instances (u250_00 ... u250_19)
- `binpack3.txt`: 500-item uniform instances (u500_00 ... u500_19)
- `binpack4.txt`: 1000-item uniform instances (u1000_00 ... u1000_19)

## Quick Start

Run from the project root (`problem1_bin_packing_aco/`):

```bash
# Default: u500_02 with FAST preset + FFD baseline
python3 quick_test.py

# Choose a specific instance
python3 quick_test.py --instance u120_00
python3 quick_test.py --instance u250_05
python3 quick_test.py --instance u1000_10

# Try different presets on the same instance
python3 quick_test.py --instance u500_02 --preset QUICK_TEST
python3 quick_test.py --instance u500_02 --preset BALANCED
python3 quick_test.py --instance u500_02 --preset INTENSIVE

# Combine instance + preset + skip baseline
python3 quick_test.py --instance u120_00 --preset INTENSIVE --no-baseline

# Save iteration logs to CSV (for plotting convergence later)
python3 quick_test.py --instance u500_02 --log-dir results/my_run
```

## Results

- **Terminal**: Summary statistics (bins used, gap from optimal, runtime, unused capacity)
- **Baseline**: FFD (First-Fit Decreasing) shown for reference
- **CSV logs**: Optional (use `--log-dir` flag) for plotting convergence

## Presets

Defined in `src/algorithm/constants.py`:

| Preset       | Ants | Iterations | α   | β   | ρ    | Use Case           |
| ------------ | ---- | ---------- | --- | --- | ---- | ------------------ |
| `QUICK_TEST` | 8    | 40         | 1.0 | 0.6 | 0.25 | Fast debugging     |
| `FAST`       | 24   | 120        | 1.1 | 0.8 | 0.18 | Default run        |
| `BALANCED`   | 32   | 180        | 1.2 | 0.7 | 0.14 | Deeper exploration |
| `INTENSIVE`  | 48   | 800        | 1.4 | 0.6 | 0.60 | Long runs          |

## File Structure
