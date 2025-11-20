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

### Instances used in this project

For the experiments and results reported in this project we used the following benchmark instances (from the uniform sets above):

- `u120_00`, `u120_06`
- `u250_00`, `u250_02`
- `u500_00`, `u500_02`
- `t249_00`, `t501_00`

## File Structure

```
problem1_bin_packing_aco/
├── README.md                     
├── requirements.txt                
├── data/
│   └── raw/
│       ├── binpack1.txt
│       ├── binpack2.txt
│       ├── binpack3.txt
│       └── binpack4.txt
├── scripts/
│   ├── download_data.py
│   ├── quick_test.py              # single-instance demo runner
│   ├── run_full_eval.py           # batch evaluation runner
    ├── run_and_save_pheromone.py  # Run and get Pheromone Heatmap       
│   └── print_solution_csv.py
├── src/
│   ├── __init__.py
│   ├── algorithm/
│   │   ├── aco.py
│   │   ├── baseline.py
│   │   ├── constants.py
│   │   ├── components/
│   │   │   ├── ant.py
│   │   │   ├── heuristic.py
│   │   │   ├── pheromone.py
│   │   │   └── ls/
│   │   │       └── ls.py
│   │   └── steps/
│   │       ├── construct.py
│   │       ├── evaluate.py
│   │       ├── initialize.py
│   │       └── update.py
│   ├── data/
│   │   └── loader.py
│   └── logging/
│       └── run_logger.py
└── results/
   └── full_evaluation/           # output from batch runs (CSV + plots)
```

## Quick Start

Run from the project root (`problem1_bin_packing_aco/`):

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Default: u500_02 with INTENSIVE preset + FFD baseline
python3 scripts/quick_test.py

# Choose a specific instance
python3 scripts/quick_test.py --instance u120_00
python3 scripts/quick_test.py --instance u250_05
python3 scripts/quick_test.py --instance u1000_10

# Try different presets on the same instance
python3 scripts/quick_test.py --instance u500_02 --preset QUICK_TEST
python3 scripts/quick_test.py --instance u500_02 --preset BALANCED
python3 scripts/quick_test.py --instance u500_02 --preset INTENSIVE

# Combine instance + preset + skip baseline
python3 scripts/quick_test.py --instance u120_00 --preset INTENSIVE --no-baseline

# Save iteration logs to CSV (for plotting convergence later)
python3 scripts/quick_test.py --instance u500_02 --log-dir results/my_run

```

## Results

- **Terminal**: Summary statistics (bins used, gap from optimal, runtime, unused capacity)
- **Baseline**: FFD (First-Fit Decreasing) shown for reference
- **CSV logs**: Optional (use `--log-dir` flag) for plotting convergence

## Presets

Defined in `src/algorithm/constants.py`:


## Batch evaluation: `run_full_eval.py`

Use `scripts/run_full_eval.py` to run batch evaluations across multiple instances and seeds. The script defaults to the `INTENSIVE` preset (can be overridden), saves per-instance CSV summaries, basic plots and a master CSV under `results/full_evaluation/`.

Key behavior:
- Default seeds: `1,2,3,4` (can be overridden)
- Default preset: `INTENSIVE` (can be overridden via the `FAST_EVAL_PRESET` or `PRESET` environment variables). The script will use the chosen preset from `src/algorithm/constants.py`.
- Local search is disabled in the evaluation mode used for quick batch runs.
- Results directory: `results/full_evaluation/<instance>/` (e.g. `results/full_evaluation/u250_00/`) contains `summary_<instance>.csv`, `baseline.csv`, a status file `<instance>.status`, and a `plots/` folder with PNGs.

Examples (run from project root `problem1_bin_packing_aco/`):

```bash
# Run only instance u250_00 with seeds 1..4 (INTENSIVE preset):
FAST_INSTANCES=u250_00 FAST_EVAL_SEEDS=1,2,3,4 PRESET=INTENSIVE python3 scripts/run_full_eval.py

# Run multiple instances (comma separated):
FAST_INSTANCES=u120_00,u250_00 FAST_EVAL_SEEDS=10,11,12,13 python3 scripts/run_full_eval.py

# Use default behavior (a fixed desired instance list inside the script and seeds 1..4):
python3 scripts/run_full_eval.py
```

Notes:
- The script writes a PID file while running (`results/run_full_eval.pid`) to help with monitoring; you can safely remove PID or `.status` files only after confirming no process is running for that PID.
- A master summary CSV is written to `results/full_evaluation/master_summary.csv` after runs finish.

**For the examiner (quick checklist)**

- **Run environment**: Python 3.8+ recommended. From the project root run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- **Run a quick demo** (default instance `u500_02`):

```bash
python3 scripts/quick_test.py
```

- **Run a specific instance** (example):

```bash
python3 scripts/quick_test.py --instance u120_00 --preset FAST
```

- **If data is missing**: the repo includes `data/raw/` files. If the examiner prefers to re-download the OR-Library benchmarks, run:

```bash
python3 scripts/download_data.py
```

- **If you need the full batch evaluation**: use `scripts/run_full_eval.py` with the environment variables shown above.

