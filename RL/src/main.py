import argparse
from pathlib import Path
from config import load_config
from train import train_loop
from eval import evaluate_checkpoint
from plots import plot_learning_curves, plot_success_rate
from baselines import run_random_baseline, run_heuristic_baseline
from utils import save_manifest, set_global_seeds

def main():
    parser = argparse.ArgumentParser(prog="rl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- TRAIN ----
    p_train = sub.add_parser("train", help="Train Q-learning agent")
    p_train.add_argument("--map", choices=["4x4", "6x6", "8x8"], default="8x8")
    p_train.add_argument("--episodes", type=int, default=20000)
    p_train.add_argument("--epsilon_schedule", choices=["exp", "linear", "two_phase"], default="exp")
    p_train.add_argument("--alpha", type=float, default=0.1)
    p_train.add_argument("--alpha_schedule", choices=["const", "linear", "inv"], default="const")
    p_train.add_argument("--alpha_min", type=float, default=0.1)
    p_train.add_argument("--gamma", type=float, default=0.99)
    p_train.add_argument("--optimistic_init", type=float, default=0.0)
    p_train.add_argument("--step_penalty", type=float, default=0.0)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--run_name", type=str, default=None)
    p_train.add_argument("--results_root", type=Path, default=Path("Results"))

    # ---- EVAL ----
    p_eval = sub.add_parser("eval", help="Evaluate a saved Q-table (greedy policy)")
    p_eval.add_argument("--map", choices=["4x4", "6x6", "8x8"], default="8x8")
    p_eval.add_argument("--checkpoint", type=Path, required=True)
    p_eval.add_argument("--episodes", type=int, default=1000)
    p_eval.add_argument("--seed", type=int, default=123)
    p_eval.add_argument("--results_root", type=Path, default=Path("Results"))


    # ---- BASELINES ----
    p_rand = sub.add_parser("baseline-random", help="Random policy baseline")
    p_rand.add_argument("--map", choices=["4x4", "6x6", "8x8"], default="8x8")
    p_rand.add_argument("--episodes", type=int, default=5000)
    p_rand.add_argument("--seed", type=int, default=0)
    p_rand.add_argument("--results_root", type=Path, default=Path("Results"))
    p_rand.add_argument("--use_run_desc", type=Path, default=None)

    p_heur = sub.add_parser("baseline-heuristic", help="Heuristic shortest-route attempt")
    p_heur.add_argument("--map", choices=["4x4", "6x6", "8x8"], default="8x8")
    p_heur.add_argument("--episodes", type=int, default=5000)
    p_heur.add_argument("--seed", type=int, default=0)
    p_heur.add_argument("--results_root", type=Path, default=Path("Results"))
    p_heur.add_argument("--use_run_desc", type=Path, default=None)

    # ---- PLOT ----
    p_plot = sub.add_parser("plot", help="Plot learning curves/success rates for a run")
    p_plot.add_argument("--run_dir", type=Path, required=True)
    p_plot.add_argument("--results_root", type=Path, default=Path("Results"))

    args = parser.parse_args()
    cfg = load_config(args)
    set_global_seeds(getattr(cfg, "seed", 42))

    if args.cmd == "train":
        run_dir = train_loop(cfg)
        save_manifest(cfg, run_dir)
        print(f"[OK] Trained. Artifacts in: {run_dir}")

    elif args.cmd == "eval":
        stats = evaluate_checkpoint(cfg, args.checkpoint, args.episodes)
        print(stats)

    elif args.cmd == "baseline-random":
        stats = run_random_baseline(cfg, args.episodes)
        print(stats)

    elif args.cmd == "baseline-heuristic":
        stats = run_heuristic_baseline(cfg, args.episodes)
        print(stats)

    elif args.cmd == "plot":
        plot_learning_curves(args.run_dir)
        plot_success_rate(args.run_dir)
        print("[OK] Plots saved to reports/ within run dir")

if __name__ == "__main__":
    main()
