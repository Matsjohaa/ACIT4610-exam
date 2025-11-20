# ACIT4610 Final Project – PSO Implementation

## Overview
This branch implements Particle Swarm Optimization (PSO) for Problem 2 of the ACIT4610 Final Project (2025).  
It evaluates PSO performance on four benchmark functions:

- Sphere  
- Rosenbrock  
- Rastrigin  
- Ackley  

Each function is tested for dimensions n = 2, 10, and 30, with 30 independent runs per setting.  
Both gbest (global best) and lbest (local best) topologies are implemented and compared.

---

## Project Structure
```
PSO/
├─ README.md                         # project documentation
│
├─ notebooks/
│   └─ PSO_comparison.ipynb          # visuals + baseline vs strong comparison
│
├─ results/
│   ├─ baseline/                     # baseline preset outputs
│   │   ├─ runs_gbest.csv
│   │   ├─ runs_lbest.csv
│   │   ├─ summary_gbest.csv
│   │   ├─ summary_lbest.csv
│   │   ├─ boxplot_gbest.png
│   │   ├─ boxplot_lbest.png
│   │   ├─ curves_gbest/*.npy
│   │   └─ curves_lbest/*.npy
│   │
│   └─ strong/                       # strong preset outputs
│       ├─ runs_gbest.csv
│       ├─ runs_lbest.csv
│       ├─ summary_gbest.csv
│       ├─ summary_lbest.csv
│       ├─ boxplot_gbest.png
│       ├─ boxplot_lbest.png
│       ├─ curves_gbest/*.npy
│       └─ curves_lbest/*.npy
│
└─ src/
    ├─ config.py                     # PSOParams defaults & success thresholds
    ├─ core.py                       # PSO engine (gbest/lbest, early stopping)
    ├─ experiment.py                 # run_suite() with param_factory + CSV logging
    ├─ functions.py                  # benchmark test functions
    ├─ topologies.py                 # neighborhood structures (gbest/lbest)
    └─ run_grid.py                   # main CLI for full experiment runs

```

---

## 1. Installation

1. **Requirements**
   ```bash
   pip install -r requirements.txt
   ```

## 2. How to Run the Code

Run from the project root (PSO):
```bash
# For baseline preset:
python3 src/run_grid.py \
  --preset baseline \
  --topologies both \
  --runs 30 \
  --dims 2 10 30 \
  --seed 123 \
  --outdir results/baseline

# For strong preset:
python3 src/run_grid.py \
  --preset strong \
  --topologies both \
  --runs 30 \
  --dims 2 10 30 \
  --seed 123 \
  --outdir results/strong
```

Expected output:
```
results/<preset>/
  runs_gbest.csv          summary_gbest.csv          boxplot_gbest.png
  runs_lbest.csv          summary_lbest.csv          boxplot_lbest.png
  curves_gbest/*.npy      curves_lbest/*.npy
```
For the full overview of CLI commands, see run_grid.py

---

## 3. Visualisations
Open the comparison notebook from the project root:
```bash
jupyter notebook notebooks/PSO_comparison.ipynb
```

The notebook renders:

- Final-fitness boxplots (30 runs) per topology, for baseline and strong, plus side-by-side comparison.

- Convergence curves (mean ± std) at n = 10 for each function, overlaying baseline vs strong for gbest and lbest.

- Success-rate tables from summary_*.csv.


