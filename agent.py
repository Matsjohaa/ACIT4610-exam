"""
Tabular Q-learning agent with ε-greedy exploration.
Meets: Q-table init zeros; α, γ; ε schedule (linear/exp). :contentReference[oaicite:7]{index=7}
"""
import numpy as np

class QTableAgent:
    def __init__(self, n_states: int, n_actions: int, alpha: float, gamma: float,
                 epsilon_schedule: str, episodes: int, optimistic_init: float = 0.0):
        self.Q = np.full((n_states, n_actions), 0.1, dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.schedule = epsilon_schedule
        self.episodes = episodes
        init_val = float(optimistic_init)
        if init_val > 0.0:
            self.Q = np.full((n_states, n_actions), init_val, dtype=np.float32)
        else:
            self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def epsilon(self, episode: int) -> float:
        if self.schedule == "linear":
            return max(0.10, 1.0 - episode / self.episodes)
        return max(0.10, 0.9995 ** episode)

    def act(self, state: int, episode: int, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon(episode):
            return int(rng.integers(0, self.Q.shape[1]))  # cast to built-in int
        return int(np.argmax(self.Q[state]))  # already cast (keep it)

    def update(self, s, a, r, s_next, done):
        best_next = 0.0 if done else np.max(self.Q[s_next])
        td_target = r + self.gamma * best_next
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])
