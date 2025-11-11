"""
Training loop for tabular Q-learning on FrozenLake-v1.

Responsibilities:
  • Build the environment (optionally with step shaping), save the exact map layout (desc.npy),
    and export a grid preview image.
  • Run long training with ε-greedy exploration and scheduled learning rate α.
  • Log per-episode returns and success flags; checkpoint Q at intervals.
  • Export tidy CSV/JSON artifacts under Results/ for downstream notebooks.
"""

import json
import csv
from datetime import datetime
from gymnasium.spaces import Discrete
from env import make_env, visualize_grid
from agent import QTableAgent
from utils import ensure_dirs, save_checkpoint, RunningStats
import numpy as np
import hashlib


def _export_tidy_results(cfg, returns, successes):
    """
    Write a tidy per-episode dataset and a compact run manifest to Results/.
    Produces:
      • Results/rl_train_<map>_<run_name>.csv with columns:
        [run_name, map, episode, return, success, epsilon, alpha, gamma, seed, timestamp]
      • Results/rl_train_<map>_<run_name>.json with high-level run metadata.

    Both ε (exploration rate) and α (learning rate) are re-derived
    deterministically for each episode using their schedules.
    """
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    out = cfg.results_root / f"rl_train_{cfg.env_map}_{cfg.run_name}.csv"

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run_name", "map", "episode", "return", "success",
            "epsilon", "alpha", "gamma", "seed", "timestamp"
        ])

        for ep, (ret, suc) in enumerate(zip(returns, successes), start=1):
            ep_idx = ep - 1  # zero-based index for schedules

            # --- epsilon schedule ---
            if cfg.epsilon_schedule == "linear":
                eps = max(0.10, 1.0 - ep_idx / cfg.episodes)
            elif cfg.epsilon_schedule == "two_phase":
                split = int(0.5 * cfg.episodes)
                if ep_idx < split:
                    eps = max(0.20, 1.0 - 0.5 * ep_idx / max(1, split))
                else:
                    eps = max(0.10, 0.995 ** (ep_idx - split))
            else:  # default exponential
                eps = max(0.10, 0.9995 ** ep_idx)

            # --- alpha schedule ---
            if getattr(cfg, "alpha_schedule", "const") == "linear":
                frac = 1.0 - (ep_idx / max(1, cfg.episodes))
                alpha_ep = max(cfg.alpha_min, cfg.alpha * frac)
            elif getattr(cfg, "alpha_schedule", "const") == "inv":
                k = max(1.0, 0.1 * cfg.episodes)
                alpha_ep = max(cfg.alpha_min, cfg.alpha / (1.0 + ep_idx / k))
            else:  # constant (default)
                alpha_ep = cfg.alpha

            w.writerow([
                cfg.run_name, cfg.env_map, ep, float(ret), int(suc),
                eps, alpha_ep, cfg.gamma, cfg.seed, datetime.now().isoformat()
            ])

    # Compact manifest for quick reference
    man = {
        "run_name": cfg.run_name,
        "map": cfg.env_map,
        "episodes": cfg.episodes,
        "alpha": cfg.alpha,
        "alpha_schedule": getattr(cfg, "alpha_schedule", "const"),
        "alpha_min": getattr(cfg, "alpha_min", 0.1),
        "gamma": cfg.gamma,
        "epsilon_schedule": cfg.epsilon_schedule,
        "seed": cfg.seed,
        "optimistic_init": getattr(cfg, "optimistic_init", 0.0),
        "step_penalty": getattr(cfg, "step_penalty", 0.0),
    }

    with open(cfg.results_root / f"rl_train_{cfg.env_map}_{cfg.run_name}.json", "w") as jf:
        json.dump(man, jf, indent=2)



def train_loop(cfg):
    """
    Run the Q-learning training loop and persist all artifacts.
    Steps:
      1) Prepare directories; build env with optional step penalty; save desc.npy and grid PNG.
      2) Construct QTableAgent with α/γ/ε settings; run ε-greedy episodes.
      3) Every K episodes, checkpoint Q and append a JSON log line with summary stats.
      4) At the end, save final Q, returns.csv, successes.csv, and tidy Results/ files.
    Returns:
        Path to the run directory (cfg.run_dir).
    """
    # --- setup & dirs ---
    ret_stats = RunningStats()
    run_dir, reports_dir, ckpt_dir = cfg.run_dir, cfg.reports_dir, cfg.checkpoints_dir
    ensure_dirs(run_dir, reports_dir, ckpt_dir)

    env = make_env(cfg.env_map, cfg.seed, step_penalty=cfg.step_penalty)

    # Save the exact map layout used for this run
    desc = env.unwrapped.desc  # bytes array
    np.save(cfg.run_dir / "desc.npy", desc)
    checksum = hashlib.md5(desc.tobytes()).hexdigest()
    with open(cfg.run_dir / "map_info.json", "w") as f:
        json.dump({"map": cfg.env_map, "desc_md5": checksum}, f, indent=2)

    # Static visualization for the report
    visualize_grid(env, reports_dir / f"grid-{cfg.env_map}.png")

    # --- type-safe spaces ---
    obs_space, act_space = env.observation_space, env.action_space
    if not (isinstance(obs_space, Discrete) and isinstance(act_space, Discrete)):
        raise TypeError("FrozenLake-v1 should have Discrete observation and action spaces.")
    n_states, n_actions = obs_space.n, act_space.n

    # --- agent, rng, logs ---
    agent = QTableAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon_schedule=cfg.epsilon_schedule,
        episodes=cfg.episodes,
        optimistic_init=getattr(cfg, "optimistic_init", 0.0),
        alpha_schedule=getattr(cfg, "alpha_schedule", "const"),
        alpha_min=getattr(cfg, "alpha_min", 0.1),
    )
    rng = np.random.default_rng(cfg.seed)
    returns, successes = [], []
    log_path = run_dir / "train_log.jsonl"
    CHECKPOINT_EVERY = 1000

    try:
        for ep in range(cfg.episodes):
            # No per-episode seed (let env RNG evolve)
            s, _ = env.reset()
            done = False
            ep_return = 0.0

            # Occasional debug of schedules
            if (ep + 1) in (1, 10, 100, 1000, 5000):
                print(f"[DBG] episode={ep+1} epsilon={agent.epsilon(ep):.3f}")

            while not done:
                a = agent.act(s, ep, rng)  # ε-greedy
                s_next, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                agent.update(s, a, r, s_next, done, episode=ep)  # Q-learning update with α schedule
                s = s_next
                ep_return += r

            # Episode bookkeeping
            success = int(ep_return > 0.0)
            returns.append(ep_return)
            successes.append(success)
            ret_stats.update(ep_return)

            # Periodic checkpoint + lightweight log line
            if (ep + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(agent.Q, ckpt_dir / f"qtable-ep{ep+1}.npz")
                wnd = min(CHECKPOINT_EVERY, len(successes))
                recent_sr = float(np.mean(successes[-wnd:])) if wnd > 0 else 0.0

                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "episode": ep + 1,
                        "epsilon": round(agent.epsilon(ep), 6),
                        "mean_return": ret_stats.mean,
                        "std_return": ret_stats.std,
                        "last_return": ep_return,
                        "last_success": success,
                        "recent_success_rate": recent_sr,
                    }) + "\n")

                print(
                    f"[TRAIN] ep={ep+1} | eps={agent.epsilon(ep):.3f} | "
                    f"mean_return={ret_stats.mean:.4f} ± {ret_stats.std:.4f} | "
                    f"recent_success_rate≈{recent_sr:.3f}"
                )

    except KeyboardInterrupt:
        print("[TRAIN] Interrupted — saving current checkpoint...")

    # --- final artifacts ---
    save_checkpoint(agent.Q, ckpt_dir / "qtable-final.npz")
    np.savetxt(run_dir / "returns.csv", np.array(returns), delimiter=",")
    np.savetxt(run_dir / "successes.csv", np.array(successes), delimiter=",")
    _export_tidy_results(cfg, returns, successes)
    env.close()
    return run_dir
