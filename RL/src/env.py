"""
Environment factory + visualization helpers.
Covers: instantiate FrozenLake-v1 (slippery=True), print state/action spaces,
visualize S/G/H grid, and optional per-step penalty shaping.
"""

from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque

# Tiles the agent can traverse during a path check (Start, Frozen, Goal)
PASSABLE = {b'S', b'F', b'G'}

# Supported map names and sizes (rows, cols)
MAP_SPECS = {"4x4": (4, 4), "6x6": (6, 6), "8x8": (8, 8)}


def _has_path(desc: list[list[bytes]]) -> bool:
    """
    Return True if there's a passable path S -> G using 4-neighborhood BFS.

    Parameters
    ----------
    desc : list[list[bytes]]
        FrozenLake layout (bytes) as produced by Gymnasium (e.g., [[b'S', b'F', ...], ...]).

    Notes
    -----
    - Only S, F, and G are traversable; H (holes) are blocked.
    - This is a fast solvability pre-check for random 6x6 maps.
    """
    n = len(desc)

    # Locate S and G
    start = goal = None
    for r in range(n):
        for c in range(n):
            if desc[r][c] == b'S':
                start = (r, c)
            if desc[r][c] == b'G':
                goal = (r, c)
    if start is None or goal is None:
        return False

    # Standard BFS over grid (4 directions)
    q = deque([start])
    seen = {start}
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                if desc[nr][nc] in PASSABLE and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return False


def make_env(
    map_name: str,
    seed: int,
    step_penalty: float = 0.0,
    desc=None,
    ensure_solvable: bool = True,
    p: float = 0.8,
    max_tries: int = 200
):
    """
    Create a FrozenLake-v1 environment with slippery dynamics.

    Parameters
    ----------
    map_name : {"4x4","6x6","8x8"}
        Named map size. "6x6" is generated randomly; others use built-ins.
    seed : int
        RNG seed for env.reset(seed=...).
    step_penalty : float, default 0.0
        If < 0.0, wrap env to add a negative reward for every non-terminal step (reward shaping).
    desc : array-like or None
        If provided, use this exact layout (overrides map_name generation).
    ensure_solvable : bool, default True
        For random 6x6: regenerate until S→G path exists (avoid zero-success runs).
    p : float, default 0.8
        Probability of Frozen tiles in random map generation (higher → fewer holes).
    max_tries : int, default 200
        Safety cap for regeneration loop.

    Returns
    -------
    env : gym.Env
        Configured Gymnasium environment, optionally wrapped with StepPenaltyWrapper.
    """
    if desc is not None:  # Use exact saved layout (training/eval consistency)
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None, desc=desc)
        summarize_spaces(env)

    elif map_name in {"4x4", "8x8"}:
        # Built-in maps: deterministic layouts provided by Gymnasium
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None, map_name=map_name)
        summarize_spaces(env)

    else:  # "6x6": generate until solvable (or until max_tries)
        tries = 0
        while True:
            # Note: some Gym versions allow a seed here; otherwise rely on global RNG.
            generated = generate_random_map(size=MAP_SPECS[map_name][0], p=p)
            if not ensure_solvable or _has_path(generated):
                break
            tries += 1
            if tries >= max_tries:
                # Fallback: accept last generated map to avoid infinite loops
                break

        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None, desc=generated)
        summarize_spaces(env)

    # Seed the environment's RNG and initial state
    env.reset(seed=seed)

    # Optional reward shaping: add a small negative reward for non-terminal steps
    if step_penalty < 0.0:
        env = StepPenaltyWrapper(env, step_penalty=step_penalty)
    return env


def summarize_spaces(env):
    """
    Print a compact summary of observation/action spaces (and discrete sizes if available).
    """
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    try:
        nS = env.observation_space.n
        nA = env.action_space.n
        print(f"Number of states: {nS}")
        print(f"Number of actions: {nA}")
    except AttributeError:
        # Spaces may not be discrete; skip counts
        pass


def visualize_grid(env, save_path: Path, show_F: bool = False):
    """
    Save an annotated FrozenLake grid with perfectly aligned cell borders.

    - Aligned grid lines (half-integer boundaries).
    - Light, color-coded background per tile.
    - Optional hiding of 'F' labels for a cleaner look.
    """
    # env.unwrapped.desc is a bytes array; cast to str for annotation and color mapping
    desc = env.unwrapped.desc.astype(str)
    nrows, ncols = desc.shape

    # Light colors for readability in reports
    color_map = {
        "S": "#b3e6ff",  # start  - light blue
        "G": "#baffc9",  # goal   - light green
        "F": "#f9f9f9",  # frozen - near white
        "H": "#ffb3b3",  # hole   - light red
    }

    # Build an RGB image for imshow
    rgb = np.zeros((nrows, ncols, 3), dtype=float)
    for r in range(nrows):
        for c in range(ncols):
            rgb[r, c] = mcolors.to_rgb(color_map.get(desc[r, c], "#ffffff"))

    # Figure size proportional to grid, with crisp cell boundaries
    fig, ax = plt.subplots(figsize=(max(3, ncols * 0.6), max(3, nrows * 0.6)))
    ax.imshow(
        rgb,
        origin="upper",
        interpolation="none",
        extent=[-0.5, ncols - 0.5, nrows - 0.5, -0.5],  # cell centers at integers
    )

    # Place S/G/H labels at integer centers (optionally omit 'F')
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

    # Clean axes
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save and close
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


class StepPenaltyWrapper(gym.Wrapper):
    """
    Reward-shaping wrapper: adds a constant penalty to every non-terminal step.

    Parameters
    ----------
    step_penalty : float, default -0.01
        Negative value added to reward when the step does not terminate or truncate the episode.

    Notes
    -----
    - Encourages shorter trajectories (can speed up learning).
    - If too large in magnitude, it can overwhelm sparse goal rewards on large maps.
    """
    def __init__(self, env, step_penalty: float = -0.01):
        super().__init__(env)
        self.step_penalty = step_penalty

    def step(self, action):
        s_next, r, terminated, truncated, info = self.env.step(action)
        # Apply penalty only to intermediate steps (not terminal/truncated)
        if not terminated and not truncated:
            r = r + self.step_penalty
        return s_next, r, terminated, truncated, info
