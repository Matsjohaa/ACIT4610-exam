"""
Episode loop with logging, moving averages, checkpointing, and grid visualization export.
Meets: long training (e.g., 10k+), tracking returns + success rate. :contentReference[oaicite:8]{index=8}
"""
import json
import numpy as np
from gymnasium.spaces import Discrete
from env import make_env, visualize_grid
from agent import QTableAgent
from utils import ensure_dirs, save_checkpoint, RunningStats  # keep RunningStats if you added it

def train_loop(cfg):
    # --- setup & dirs ---
    ret_stats = RunningStats()
    run_dir, reports_dir, ckpt_dir = cfg.run_dir, cfg.reports_dir, cfg.checkpoints_dir
    ensure_dirs(run_dir, reports_dir, ckpt_dir)

    env = make_env(cfg.env_map, cfg.seed, step_penalty=cfg.step_penalty)
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
        optimistic_init=cfg.optimistic_init,
    )
    rng = np.random.default_rng(cfg.seed)
    returns, successes = [], []
    log_path = run_dir / "train_log.jsonl"
    CHECKPOINT_EVERY = 1000

    try:
        for ep in range(cfg.episodes):
            # no per-episode seed (let env RNG evolve)
            s, _ = env.reset()
            done = False
            ep_return = 0.0

            # optional: peek at epsilon a few times to confirm exploration
            if (ep + 1) in (1, 10, 100, 1000, 5000):
                print(f"[DBG] episode={ep+1} epsilon={agent.epsilon(ep):.3f}")

            while not done:
                a = agent.act(s, ep, rng)                 # ε-greedy
                s_next, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                agent.update(s, a, r, s_next, done)       # Q-learning
                s = s_next
                ep_return += r

            # episode bookkeeping
            success = int(ep_return > 0.0)
            returns.append(ep_return)
            successes.append(success)
            ret_stats.update(ep_return)

            # periodic checkpoint + light logging
            if (ep + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(agent.Q, ckpt_dir / f"qtable-ep{ep+1}.npz")
                # recent success over the last window (handle short prefix)
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

                print(f"[TRAIN] ep={ep+1} | eps={agent.epsilon(ep):.3f} | "
                      f"mean_return={ret_stats.mean:.4f} ± {ret_stats.std:.4f} | "
                      f"recent_success_rate≈{recent_sr:.3f}")

    except KeyboardInterrupt:
        print("[TRAIN] Interrupted — saving current checkpoint...")

    # --- final artifacts ---
    save_checkpoint(agent.Q, ckpt_dir / "qtable-final.npz")
    np.savetxt(run_dir / "returns.csv", np.array(returns), delimiter=",")
    np.savetxt(run_dir / "successes.csv", np.array(successes), delimiter=",")
    env.close()
    return run_dir


