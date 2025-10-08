from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import uuid

@dataclass
class Config:
    env_map: str
    episodes: int
    epsilon_schedule: str
    alpha: float
    gamma: float
    seed: int
    run_name: str
    optimistic_init: float
    step_penalty: float
    # paths...
    root: Path
    run_dir: Path
    reports_dir: Path
    checkpoints_dir: Path

def load_config(args) -> Config:
    root = Path(__file__).resolve().parent
    # use getattr() so missing fields don’t crash for some subcommands
    run_name = getattr(args, "run_name", None) or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    env_map = getattr(args, "map", "8x8")
    episodes = getattr(args, "episodes", 0)              # not used by baselines/eval
    epsilon_schedule = getattr(args, "epsilon_schedule", "exp")
    alpha = getattr(args, "alpha", 0.8)
    gamma = getattr(args, "gamma", 0.99)
    seed = getattr(args, "seed", 0)
    optimistic_init = getattr(args, "optimistic_init", 0.0)
    step_penalty = getattr(args, "step_penalty", 0.0)

    run_dir = root / "runs" / run_name
    return Config(
        env_map=env_map,
        episodes=episodes,
        epsilon_schedule=epsilon_schedule,
        alpha=alpha,
        gamma=gamma,
        seed=seed,
        run_name=run_name,
        optimistic_init=optimistic_init,
        step_penalty=step_penalty,
        root=root,
        run_dir=run_dir,
        reports_dir=run_dir / "reports",
        checkpoints_dir=run_dir / "checkpoints",
    )
