# ACIT4610 Final Project — Tabular Q-Learning on FrozenLake

This repository contains a tabular Q-learning implementation for the Gymnasium FrozenLake-v1 environment.  
It includes dynamic ε-greedy exploration, adjustable α and γ, random/heuristic baselines, visualizations, and evaluation.

---

## Repository Structure

```text
RL/
  ├── agent.py              # Q-table agent (act / update / save / load)
  ├── baselines.py          # Random and heuristic (shortest-path) baselines
  ├── config.py             # Loads all hyperparameters and paths
  ├── env.py                # Environment factory, seeding, visualization
  ├── eval.py               # Greedy policy evaluation
  ├── main.py               # CLI entry point: train / eval / plot / baselines
  ├── plots.py              # Plots learning curves and success-rate trends
  ├── train.py              # Training loop with logging and checkpoints
  ├── utils.py              # Helper functions: reproducibility, IO, CI calc
  ├── runs/                 # Auto-generated experiments, checkpoints, logs
  └── results/              # Summary JSON/CSV outputs
```

---

## Installation

**Go to RL/src folder**
```bash
   cd RL
```
```bash
   cd src
```

**Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```


### Train an agent
```bash
python -m main train --map 8x8 --episodes 80000 --epsilon_schedule exp --alpha 0.8 --gamma 0.99 --seed 42 --run_name 8x8_ref
```

### Evaluate a saved Q-table
```bash
python -m main eval --map 8x8 --checkpoint runs/8x8_ref/checkpoints/qtable-final.npz --episodes 10000
```

### Plot learning curves and success rates
```bash
python -m main plot --run_dir runs/8x8_ref
```

### Run baselines
```bash
# Random
python -m main baseline-random --map 8x8 --episodes 80000 --use_run_desc runs/8x8_ref
# Heuristic (shortest path, ignores slippage)
python -m main baseline-heuristic --map 8x8 --episodes 20000 --use_run_desc runs/8x8_ref
```

---

## Key Features

- Environment introspection & grid visualization (S / G / H).
- Tabular Q-learning with tunable `α`, `γ`, and ε-schedules (`linear`, `exp`, `two_phase`).  
- Long-run training with moving averages of rewards and success rate.  
- Map comparison: 6×6, and 8×8 layouts.  
- Baselines: random policy and deterministic shortest-path heuristic.  
- Evaluation: greedy success rate (`ε=0`), average return, and plots for each run.  
- Reproducibility: each run stored under `runs/<run-name>/` with seeds and `desc.npy` map.

---

## Experiment Reproduction

| Category | Parameter varied | Command examples |
|-----------|------------------|------------------|
| ε schedules | `--epsilon_schedule` = `exp` / `linear` | Train three 8×8 agents with same α,γ |
| γ (discount) | `--gamma` = `0.99` vs `0.95` | Compare far- vs near-sighted agents |
| α (learning rate) | `--alpha_schedule` = `const` / `linear` / `inv` | Compare stability vs speed |
| Baselines | Random / Heuristic | Fixed reference lines |
| **Reward shaping** | `--step_penalty` = `0.0`, `-0.001`, `-0.01` | Test impact of mild vs strong step penalty on 6×6 and 8×8 maps |

After training, evaluate each Q-table:
```bash
python -m main eval --map 8x8 --checkpoint runs/<run_name>/checkpoints/qtable-final.npz --episodes 20000
```

## Reproducibility notes

- Use the same seed for fair comparisons.  
- Every run saves:
  - `desc.npy` (exact map layout)
  - `manifest.json` (hyperparameters)
  - `returns.csv` and `successes.csv`
  - Plots under `reports/`
- All evaluation and baseline outputs appear under `Results/` as JSON for notebook analysis.

---

## License / Credits
Project for **ACIT4610 — Artificial Intelligence Methods**  
OsloMet, 2025.  
All code by *Kristina Kufaas* unless otherwise noted.  
Gymnasium environment © OpenAI / Farama Foundation.
