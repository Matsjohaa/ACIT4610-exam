import numpy as np

class QTableAgent:
    """
    Tabular Q-learning agent with scheduled epsilon (exploration) and alpha (learning rate).

    Supported schedules:
      - epsilon: "exp" (default), "linear", "two_phase"
      - alpha:   "const" (default), "linear", "inv"
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon_schedule: str,
        episodes: int,
        optimistic_init: float = 0.0,
        alpha_schedule: str = "const",
        alpha_min: float = 0.1,
    ):
        # Q init (optimistic or zeros)
        init_val = float(optimistic_init)
        if init_val > 0.0:
            self.Q = np.full((n_states, n_actions), init_val, dtype=np.float32)
        else:
            self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

        # Params
        self.nS = int(n_states)
        self.nA = int(n_actions)
        self.gamma = float(gamma)
        self.schedule = str(epsilon_schedule)      # epsilon schedule selector
        self.episodes = int(episodes)

        # Alpha scheduling
        self.alpha0 = float(alpha)
        self.alpha_min = float(alpha_min)
        self.alpha_schedule = str(alpha_schedule)

        # For "two_phase" epsilon, you can tweak the split ratio if desired
        self._two_phase_split = 0.5  # first half explore more, second half decay faster

    # ---------------------------
    # Epsilon schedule (explore)
    # ---------------------------
    def epsilon(self, episode: int) -> float:
        """Return ε for the given episode index (0-based)."""
        if self.schedule == "linear":
            # Linear decay from 1.0 -> 0.10
            return max(0.10, 1.0 - episode / self.episodes)

        if self.schedule == "two_phase":
            # Two-phase: flat-ish early, then exponential late
            split = int(self._two_phase_split * self.episodes)
            if episode < split:
                # mild linear decay in the first phase
                return max(0.20, 1.0 - 0.5 * episode / max(1, split))
            # faster exponential decay in the second phase
            t = episode - split
            return max(0.10, 0.995 ** t)

        # Default: exponential decay
        return max(0.10, 0.9995 ** episode)

    # ---------------------------
    # Alpha schedule (learn rate)
    # ---------------------------
    def alpha_at(self, episode: int) -> float:
        """Return α_t for the given episode index (0-based)."""
        sched = self.alpha_schedule

        if sched == "linear":
            # Linear decay from alpha0 -> alpha_min across the run
            frac = 1.0 - (episode / max(1, self.episodes))
            return max(self.alpha_min, self.alpha0 * frac)

        if sched == "inv":
            # Inverse schedule: α_t = α0 / (1 + episode / k)
            # Choose k as a fraction of total episodes so α approaches alpha_min near the end.
            k = max(1.0, 0.1 * self.episodes)  # tune factor (0.1 * episodes works well)
            return max(self.alpha_min, self.alpha0 / (1.0 + episode / k))

        # Default: constant alpha
        return self.alpha0

    # ---------------------------
    # Policy and update
    # ---------------------------
    def act(self, state: int, episode: int, rng: np.random.Generator) -> int:
        """ε-greedy action selection."""
        if rng.random() < self.epsilon(episode):
            return int(rng.integers(0, self.nA))
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool, episode: int):
        """Standard Q-learning update with scheduled alpha."""
        alpha_t = self.alpha_at(episode)
        qsa = self.Q[s, a]
        if done:
            target = r
        else:
            target = r + self.gamma * float(np.max(self.Q[s_next]))
        self.Q[s, a] = qsa + alpha_t * (target - qsa)
