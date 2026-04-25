"""
engine/q_agent.py — Agent Q-learning tabulaire pour Pi*.

Q-table : (N_STATES x N_ACTIONS) = 400 x 3
Politique : epsilon-greedy avec decay lineaire par episode.
"""

import numpy as np
import pickle
from engine.rl_env import N_STATES, N_ACTIONS


class QAgent:
    """
    Agent Q-learning tabulaire.

    alpha         : taux d'apprentissage
    gamma         : facteur d'actualisation
    epsilon       : exploration initiale -> epsilon_min via decay lineaire
    epsilon_decay : decrement par episode
    """

    def __init__(
        self,
        alpha:         float = 0.1,
        gamma:         float = 0.95,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.05,
        epsilon_decay: float = 0.001,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table       = np.zeros((N_STATES, N_ACTIONS))
        self._episode_count = 0

    # ── Politique ─────────────────────────────────────────────

    def act(self, state: int, training: bool = True,
            mask: np.ndarray = None) -> int:
        if mask is None:
            allowed = None
        elif mask.ndim == 1:
            allowed = mask          # déjà (N_ACTIONS,) — pré-indexé
        else:
            allowed = mask[state]   # (N_STATES, N_ACTIONS) — indexer par état

        if training and np.random.rand() < self.epsilon:
            if allowed is not None:
                choices = np.where(allowed)[0]
                return int(np.random.choice(choices)) if len(choices) > 0 else 0
            return np.random.randint(N_ACTIONS)

        q = self.q_table[state].copy()
        if allowed is not None:
            q[~allowed] = -np.inf
        return int(np.argmax(q))

    # ── Mise a jour ────────────────────────────────────────────

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        current_q = self.q_table[state, action]
        target_q  = reward + (0.0 if done else self.gamma * np.max(self.q_table[next_state]))
        self.q_table[state, action] += self.alpha * (target_q - current_q)

    def update_mc(self, state: int, action: int, G: float):
        """Monte Carlo update : assigne le retour realise G directement a l'etat d'entree."""
        self.q_table[state, action] += self.alpha * (G - self.q_table[state, action])

    def end_episode(self):
        self._episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    # ── Persistance ───────────────────────────────────────────

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "q_table":       self.q_table,
                "epsilon":       self.epsilon,
                "epsilon_min":   self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "episode_count": self._episode_count,
                "alpha":         self.alpha,
                "gamma":         self.gamma,
            }, f)
        print(f"[q_agent] Q-table sauvegardee : {path} ({self._episode_count} episodes)")

    @classmethod
    def load(cls, path: str) -> "QAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            alpha         = data["alpha"],
            gamma         = data["gamma"],
            epsilon_min   = data.get("epsilon_min",   0.05),
            epsilon_decay = data.get("epsilon_decay", 0.001),
        )
        agent.q_table        = data["q_table"]
        agent.epsilon        = data["epsilon"]
        agent._episode_count = data["episode_count"]
        print(f"[q_agent] Q-table chargee : {path} ({agent._episode_count} episodes)")
        return agent

    # ── Diagnostics ────────────────────────────────────────────

    def policy_summary(self) -> dict:
        """Action dominante par etat (sur les etats visites uniquement)."""
        visited = self.q_table.sum(axis=1) != 0
        actions = np.argmax(self.q_table, axis=1)
        counts  = {0: 0, 1: 0, 2: 0}
        for s in np.where(visited)[0]:
            counts[int(actions[s])] += 1
        return {
            "FLAT":  counts[0],
            "LONG":  counts[1],
            "SHORT": counts[2],
            "visited_states": int(visited.sum()),
        }
