"""
Environment factory + visualization helpers.
Covers: instantiate FrozenLake-v1 (slippery=True), print state/action spaces,
visualize S/G/H grid. :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
"""
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt

MAP_SPECS = {
    "4x4": (4, 4),
    "6x6": (6, 6),
    "8x8": (8, 8),
}

def make_env(map_name: str, seed: int, step_penalty: float = 0.0):
    assert map_name in MAP_SPECS
    nrows, ncols = MAP_SPECS[map_name]
    assert nrows == ncols, "FrozenLake expects a square map"

    # Built-in solvable maps for 4x4 and 8x8; generate desc for custom sizes (e.g., 6x6)
    if map_name in {"4x4", "8x8"}:
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None, map_name=map_name)
    else:
        desc = generate_random_map(size=nrows, p=0.8)
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None, desc=desc)

    # Seed once here; do NOT reseed each episode during training
    env.reset(seed=seed)

    # Optional reward shaping
    if step_penalty < 0.0:
        env = StepPenaltyWrapper(env, step_penalty=step_penalty)

    # Friendly info (guard for Discrete)
    if isinstance(env.observation_space, Discrete) and isinstance(env.action_space, Discrete):
        print(f"[ENV] Map={map_name}, states={env.observation_space.n}, actions={env.action_space.n}")
    else:
        print(f"[ENV] Map={map_name}, spaces: obs={env.observation_space}, act={env.action_space}")

    return env


def visualize_grid(env, save_path: Path):
    """
    Save a simple annotated grid image marking S, G, H.  Required visualization. :contentReference[oaicite:6]{index=6}
    """
    desc = env.unwrapped.desc.astype(str)
    nrows, ncols = desc.shape
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((nrows, ncols)))
    for r in range(nrows):
        for c in range(ncols):
            ax.text(c, r, desc[r, c], ha="center", va="center", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

class StepPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, step_penalty: float = -0.01):
        super().__init__(env)
        self.step_penalty = step_penalty
    def step(self, action):
        s_next, r, terminated, truncated, info = self.env.step(action)
        if not terminated and not truncated:
            r = r + self.step_penalty
        return s_next, r, terminated, truncated, info