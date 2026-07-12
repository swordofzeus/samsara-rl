from typing import Any

import numpy as np

from samsara_rl.agent import Agent
from samsara_rl.utils.history import History


class MonteCarloPrediction(Agent):
    """Every-visit Monte Carlo prediction for estimating Q(s, a).

    Generates episodes under a given policy and updates action-value
    estimates using constant-alpha MC learning. Returns are computed
    in a fully vectorized manner without explicit loops over time steps.

    Args:
        mdp: Environment with ``STATE_COUNT``, ``ACTION_COUNT``,
            ``initial_state()``, ``is_terminal_state()``, and ``step()``
            methods.
        policy: A stochastic policy array of shape
            ``(STATE_COUNT, ACTION_COUNT)`` used to sample actions.
        alpha: Learning rate for incremental Q updates.
        gamma: Discount factor.
    """

    def __init__(self, mdp: Any, policy: np.ndarray, alpha: float = 0.01, gamma: float = 1) -> None:
        super().__init__(mdp, policy, alpha, gamma)
        self.q = np.zeros((mdp.observation_space.n, mdp.action_space.n))

    def post_visit(self, history: History, terminal: bool) -> None:
        return

    def post_episode(self, history: History) -> None:
        """Update Q-table from a single episode trajectory.

        Uses advanced indexing to apply the constant-alpha MC update
        Q(s,a) <- Q(s,a) + alpha * (G - Q(s,a)) for every visited
        (state, action) pair in the trajectory.

        Args:
            history: The complete episode history.
        """
        # Compute discounted rewards from 0..N-1 excluding terminal state
        discounted_trajectory = self._discounted_cum_trajectory(history.past_rewards()[0:-2])
        s = history.past_states()[0:-2].astype(int)
        a = history.past_actions()[0:-2].astype(int)

        bellman_error = self.alpha * (discounted_trajectory - self.q[s, a])
        np.add.at(self.q, (s, a), bellman_error)

    def _discounted_cum_trajectory(self, reward: np.ndarray) -> np.ndarray:
        """Compute discounted returns for every time step, vectorized.

        Avoids the standard O(T) reverse loop by factoring out discount
        weights from a cumulative sum:

        1. Divide each reward by its positional gamma power to normalize.
        2. Reverse and cumsum so earlier states accumulate future rewards.
        3. Multiply back by gamma powers to restore correct discounting.

        Args:
            reward: Array of shape ``(T,)`` containing rewards.

        Returns:
            Array of shape ``(T,)`` with the discounted return G_t for
            each time step.
        """
        discount_ratio = self.gamma ** np.arange(0, len(reward))[::-1]
        reversed_reward = reward
        reversed_reward = reversed_reward / discount_ratio
        reversed_reward_cum = reversed_reward[::-1].cumsum()
        reversed_reward_cum = reversed_reward_cum * discount_ratio[::-1]
        result: np.ndarray = reversed_reward_cum[::-1]
        return result

    def get_q_values(self, state: Any) -> np.ndarray:
        result: np.ndarray = self.q[int(state)]
        return result
