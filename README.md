# ACIT4610-exam

## Repository Structure

```text
rl/
  ├─ main.py                 # CLI entry point: train/eval/plot
  ├─ config.py               # Hyperparameters + experiment grids
  ├─ env.py                  # Env factory, seeding, grid visualization
  ├─ agent.py                # QTableAgent (act/update/save/load)
  ├─ train.py                # Episode loop, logging, checkpointing
  ├─ eval.py                 # Greedy policy eval, success rate, returns
  ├─ baselines.py            # Random policy + heuristic “shortest route”
  ├─ plots.py                # Curves: returns, moving avg, success rate
  ├─ utils.py                # Reproducibility, meters, CSV/JSON logging
  ├─ reports/                # Auto-exported figs/tables
  └─ README.md               # How to run, reproduce, figures list

```

# RL Final Project Module — Q-learning on FrozenLake-v1

This module implements **tabular Q-learning** with **ε-greedy exploration** on **Gym/Gymnasium FrozenLake-v1** (slippery=True), with required **baselines**, **plots**, and **evaluation**.

## What this covers (per assignment)
- Env introspection & grid visualization (S/G/H). ✔️ :contentReference[oaicite:12]{index=12}
- Q-learning with α, γ; ε schedules (linear/exp). ✔️ :contentReference[oaicite:13]{index=13}
- Long training; track returns & success rate; moving averages. ✔️ :contentReference[oaicite:14]{index=14}
- Compare maps (4×4 / 6×6 / 8×8). ✔️ :contentReference[oaicite:15]{index=15}
- Baselines: random policy + shortest-route heuristic. ✔️ :contentReference[oaicite:16]{index=16}
- Evaluation: greedy policy success rate & average return. ✔️ :contentReference[oaicite:17]{index=17}
- Clean code, README, reproducibility. ✔️ :contentReference[oaicite:18]{index=18}

## Quick start
```bash
# Train (example)
python -m rl.main train --map 8x8 --episodes 20000 --epsilon_schedule exp --alpha 0.8 --gamma 0.99 --seed 42

# Plot curves for a run
python -m rl.main plot --run_dir rl/runs/<your-run-id>

# Evaluate final Q-table
python -m rl.main eval --checkpoint rl/runs/<your-run-id>/checkpoints/qtable-final.npz --episodes 1000

# Baselines
python -m rl.main baseline-random --map 8x8 --episodes 5000
python -m rl.main baseline-heuristic --map 8x8 --episodes 5000
