import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def moving_average(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

def plot_learning_curves(run_dir: Path, window: int = 1000, point_size: float = 3.0):
    returns = np.loadtxt(run_dir / "returns.csv", delimiter=",")
    n = len(returns)

    # x-coords
    x = np.arange(n)

    # Smooth curve
    ma = moving_average(returns, window)
    x_ma = np.arange(window - 1, window - 1 + len(ma))
    fig, ax = plt.subplots()

    # 1) Show successes as light points (won’t paint the whole background)
    success_idx = np.nonzero(returns > 0)[0]
    if success_idx.size:
        ax.vlines(success_idx, 0, 1, linewidth=0.5, alpha=0.05, label="Success episodes")
    else:
        ax.scatter(x, returns, s=point_size, alpha=0.3, label="Return per episode")

    # 2) Moving average
    ax.plot(x_ma, ma, linewidth=2.0, label=f"Moving avg ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    fig.savefig(run_dir / "reports" / "learning_curve.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

def plot_success_rate(run_dir: Path, window: int = 1000):
    successes = np.loadtxt(run_dir / "successes.csv", delimiter=",")
    # n = len(successes)
    # x = np.arange(n)
    ma = moving_average(successes, window)
    x_ma = np.arange(window - 1, window - 1 + len(ma))
    fig, ax = plt.subplots()
    ax.plot(x_ma, ma, linewidth=2.0, label=f"Success rate (moving avg {window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success (fraction)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    fig.savefig(run_dir / "reports" / "success_rate.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
