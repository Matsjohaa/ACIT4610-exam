"""
Environment factory + visualization helpers.
Covers: instantiate FrozenLake-v1 (slippery=True), print state/action spaces,
visualize S/G/H grid
"""
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

MAP_SPECS = {"4x4": (4,4), "6x6": (6,6), "8x8": (8,8)}

def make_env(map_name: str, seed: int, step_penalty: float = 0.0, desc=None):
    if desc is not None:  # use exact saved layout
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None, desc=desc)
        summarize_spaces(env)
    elif map_name in {"4x4", "8x8"}:
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None, map_name=map_name)
        summarize_spaces(env)
    else:  # 6x6 generated
        generated = generate_random_map(size=MAP_SPECS[map_name][0], p=0.8)
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None, desc=generated)
        summarize_spaces(env)
    env.reset(seed=seed)
    if step_penalty < 0.0:
        env = StepPenaltyWrapper(env, step_penalty=step_penalty)
    return env


def summarize_spaces(env):
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    # Optional: show counts explicitly
    try:
        nS = env.observation_space.n
        nA = env.action_space.n
        print(f"Number of states: {nS}")
        print(f"Number of actions: {nA}")
    except AttributeError:
        pass


def visualize_grid(env, save_path: Path, show_F: bool = False):
    """
    Save an annotated FrozenLake grid with perfectly aligned cell borders.

    - Aligned grid lines (drawn at half-integers).
    - Light, color-coded background per tile.
    - Optionally hide 'F' letters for a cleaner look.
    """
    desc = env.unwrapped.desc.astype(str)
    nrows, ncols = desc.shape

    # Light colors for readability
    color_map = {
        "S": "#b3e6ff",  # start - light blue
        "G": "#baffc9",  # goal  - light green
        "F": "#f9f9f9",  # frozen - near white
        "H": "#ffb3b3",  # hole   - light red
    }

    # Build RGB image
    rgb = np.zeros((nrows, ncols, 3), dtype=float)
    for r in range(nrows):
        for c in range(ncols):
            rgb[r, c] = mcolors.to_rgb(color_map.get(desc[r, c], "#ffffff"))

    # Figure sized to grid
    fig, ax = plt.subplots(figsize=(max(3, ncols * 0.6), max(3, nrows * 0.6)))

    # Place image so that each cell center is at integer (c, r)
    # and boundaries are at half-integers. This makes lines align perfectly.
    ax.imshow(
        rgb,
        origin="upper",
        interpolation="none",
        extent=[-0.5, ncols - 0.5, nrows - 0.5, -0.5],
    )

    # Annotations at integer centers
    for r in range(nrows):
        for c in range(ncols):
            ch = desc[r, c]
            if ch == "F" and not show_F:
                continue
            ax.text(
                c, r, ch,
                ha="center", va="center",
                fontsize=14, fontweight="bold", color="black"
            )

    # Draw crisp grid lines exactly on cell boundaries
    x_bounds = np.arange(-0.5, ncols, 1.0)
    y_bounds = np.arange(-0.5, nrows, 1.0)
    for x in x_bounds:
        ax.vlines(x, -0.5, nrows - 0.5, colors="black", linewidth=1.2)
    for y in y_bounds:
        ax.hlines(y, -0.5, ncols - 0.5, colors="black", linewidth=1.2)

    # Formatting
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

class StepPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, step_penalty: float = -0.01):
        super().__init__(env); self.step_penalty = step_penalty
    def step(self, action):
        s_next, r, terminated, truncated, info = self.env.step(action)
        if not terminated and not truncated:
            r = r + self.step_penalty
        return s_next, r, terminated, truncated, info