import numpy as np
from samsara_rl.utils.policy.policy_utils import sample


class MonteCarloPrediction:
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

    def __init__(self, mdp, policy, alpha=0.01, gamma=0.9):
        self.q_table = np.zeros((mdp.STATE_COUNT, mdp.ACTION_COUNT))
        self.mdp = mdp
        self.policy = policy
        self.gamma = gamma
        self.alpha = alpha

    def update_q_values(self, trajectory):
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
        self.q_table[s, a] += bellman_error

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

    def run_episode(self):
        """Generate a complete episode under the current policy.

        Samples actions from the policy and steps through the MDP until
        a terminal state is reached.

        Returns:
            Array of shape ``(T, 3)`` where each row is
            ``[state, action, reward]``.
        """
        curr_state, curr_step = self.mdp.initial_state(), 0

        trajectory = np.zeros(
            (1000, 3)
        )

        while not self.mdp.is_terminal_state(curr_state):
            sampled_action = sample(self.policy, curr_state)
            reward, next_state = self.mdp.step(
                curr_state, sampled_action
            )
            trajectory[curr_step][0] = curr_state
            trajectory[curr_step][1] = sampled_action
            trajectory[curr_step][2] = reward
            curr_step += 1
            curr_state = next_state

        return trajectory[0:curr_step]

    def evaluate(self, max_iter: int = 40000):
        """Run Monte Carlo prediction for a fixed number of episodes.

        Args:
            max_iter: Number of episodes to sample and learn from.
        """
        for _ in range(0, max_iter):
            trajectory = self.run_episode()
            self.update_q_values(trajectory)
