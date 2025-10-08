"""
Required plots: cumulative reward per episode + moving average; success-rate over training.
Save into run_dir/reports/. :contentReference[oaicite:11]{index=11}
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(run_dir: Path, window: int = 100):
    returns = np.loadtxt(run_dir / "returns.csv", delimiter=",")
    ma = moving_average(returns, window)
    fig, ax = plt.subplots()
    ax.plot(returns, label="Return per episode")
    ax.plot(np.arange(len(ma)) + window - 1, ma, label=f"Moving avg ({window})")
    ax.set_xlabel("Episode"); ax.set_ylabel("Return")
    ax.legend(); (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    fig.savefig(run_dir / "reports" / "learning_curve.png", bbox_inches="tight")
    plt.close(fig)

def plot_success_rate(run_dir: Path, window: int = 500):
    successes = np.loadtxt(run_dir / "successes.csv", delimiter=",")
    ma = moving_average(successes, window)
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(ma)) + window - 1, ma, label=f"Success rate (moving avg {window})")
    ax.set_xlabel("Episode"); ax.set_ylabel("Success (fraction)")
    ax.legend(); (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    fig.savefig(run_dir / "reports" / "success_rate.png", bbox_inches="tight")
    plt.close(fig)

def moving_average(x, w):
    if len(x) < w: return x
    return np.convolve(x, np.ones(w)/w, mode="valid")
