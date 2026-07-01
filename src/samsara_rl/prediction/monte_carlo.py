import numpy as np
from samsara_rl.prediction.policy_evaluation import PolicyEvaluation
from samsara_rl.utils.policy.policy_utils import sample


class MonteCarloPrediction(PolicyEvaluation):
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

    def __init__(self, mdp, policy, alpha=0.01, gamma=1):
        super().__init__(mdp, policy, alpha, gamma)

    def post_visit(self, trajectory):
        return

    def post_episode(self, trajectory):
        """Update Q-table from a single episode trajectory.

        Uses advanced indexing to apply the constant-alpha MC update
        Q(s,a) <- Q(s,a) + alpha * (G - Q(s,a)) for every visited
        (state, action) pair in the trajectory.

        Args:
            trajectory: Array of shape ``(T, 3)`` where each row is
                ``[state, action, reward]``.
        """
        discounted_trajectory = self._discounted_cum_trajectory(trajectory)
        trajectory = np.column_stack((trajectory, discounted_trajectory))
        s_a_pairs = trajectory[:, 0:2].astype(np.int16)
        s = s_a_pairs[:, 0]
        a = s_a_pairs[:, 1]
        bellman_error = self.alpha * (discounted_trajectory - self.q_table[s, a])
        np.add.at(self.q_table, (s, a), bellman_error)

    def _discounted_cum_trajectory(self, trajectory):
        """Compute discounted returns for every time step, vectorized.

        Avoids the standard O(T) reverse loop by factoring out discount
        weights from a cumulative sum:

        1. Divide each reward by its positional gamma power to normalize.
        2. Reverse and cumsum so earlier states accumulate future rewards.
        3. Multiply back by gamma powers to restore correct discounting.

        Args:
            trajectory: Array of shape ``(T, 3)`` where column 2 contains
                rewards.

        Returns:
            Array of shape ``(T,)`` with the discounted return G_t for
            each time step.
        """
        discount_ratio = self.gamma ** np.arange(0, len(trajectory[:, 2]))[::-1]
        reversed_reward = trajectory[:, 2]
        reversed_reward = reversed_reward / discount_ratio
        reversed_reward_cum = reversed_reward[::-1].cumsum()
        reversed_reward_cum = reversed_reward_cum * discount_ratio[::-1]
        return reversed_reward_cum[::-1]
