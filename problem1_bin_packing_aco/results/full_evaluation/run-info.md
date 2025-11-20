Run summary
===========

This folder contains results produced by running the project's Intensive preset experiments. The runs were executed with the project's `scripts/run_full_eval.py` runner and follow the same reproducible procedure used for the exam.

- Instances run: `t249_00`, `t501_00`, `u120_00`, `u120_06`, `u250_00`, `u250_02`, `u500_00`, `u500_02`.
- Seeds: `1,2,3,4` for each instance.
- Preset: `Intensive` (intensive experimental preset defined in `src/algorithm/constants.py`).
- Minimal logging: the runner was invoked with `--minimal_log` to reduce per-iteration logging (the runner sets `solver.log_every = 0` in this mode).

How the runs were launched
-------------------------

Each instance was started with environment variables to control the runner and run in the background. Example invocation used for each instance (executed from the repository root `problem1_bin_packing_aco`):

```
# Use PRESET=INTENSIVE (or FAST_EVAL_PRESET) to select the preset.
FAST_EVAL_SEEDS=1,2,3,4 FAST_INSTANCES=<instance> PRESET=INTENSIVE \
	nohup python3 scripts/run_full_eval.py --minimal_log > run_<instance>.log 2>&1 &
echo $! > run_pids/<instance>.pid
```

Notes about outputs
-------------------

- Per-instance status: the runner writes `STARTED` to `results/full_evaluation/<instance>/<instance>.status` when the instance begins and `FINISHED` when all seeds complete.
- Summary CSV: after all seeds for an instance complete the runner writes `summary_<instance>.csv` into `results/full_evaluation/<instance>/` (contains per-seed rows and runtimes).
- Logs: runtime logs are saved as `run_<instance>.log` in the project root; we also kept PID files in `run_pids/<instance>.pid` for process management.
- Plots: convergence and distribution plots are produced under `results/full_evaluation/<instance>/plots/` by the runner.

Key parameters (INTENSIVE preset)
---------------------------

- `n_ants = 90`
- `n_iterations = 200`
- `alpha = 2.0` (pheromone importance)
- `beta = 1.0` (heuristic/tightness importance)
- `rho = 0.02` (evaporation rate)
- `Q = 10.0` (pheromone deposit multiplier)

Implementation notes
--------------------

- The project's ACO implementation computes ant scores using pheromone^alpha and tightness^beta. The heuristic returns both `size` and `tightness`, but the solver currently uses only the `tightness` component (i.e. the `size` term is unused unless the code is changed).
- A deprecation warning about `datetime.utcnow()` may appear in logs; this is harmless and does not affect runs.

Sleep / machine policy
----------------------

To keep the runs running overnight on macOS the team used `caffeinate` waiters, one per run PID:

```
caffeinate -w $(cat run_pids/t249_00.pid) &
caffeinate -w $(cat run_pids/t501_00.pid) &
caffeinate -w $(cat run_pids/u120_06.pid) &
```

This prevents idle sleep while each target process is running. Note `caffeinate` does not prevent shutdowns or forced user logouts.

Where to look for results
-------------------------

All produced outputs described above live under `results/full_evaluation/<instance>/`. Check these files when runs finish:

- `results/full_evaluation/<instance>/<instance>.status`
- `results/full_evaluation/<instance>/summary_<instance>.csv`
- `results/full_evaluation/<instance>/plots/`
- `run_<instance>.log`

If you want, I can now collect the `summary_*.csv` files and produce a short table of runtimes and best-found bin counts for inclusion in the report.

