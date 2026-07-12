from typing import Any

import numpy as np

from samsara_rl.planning.planning import Planning


class PolicyIteration(Planning):
    """
    Policy Iteration for finite, discrete MDPs.

    Alternates between two steps until the policy stops changing:
      1. Policy Evaluation  — compute V(s) for the current policy via iterative
                              application of the Bellman expectation equation.
      2. Policy Improvement — update the policy greedily with respect to V(s).

    Follows the formulation in David Silver's RL Lecture 3 (Planning by DP).

    Attributes:
        mdp: The MDP to solve. Must expose STATE_COUNT, ACTION_COUNT,
             state_action_transition_matrix, and reward_matrix.
        bellman_tolerance: Convergence threshold for policy evaluation.
        policy: Current policy as a (STATE_COUNT, ACTION_COUNT) probability
                matrix. Initialised to the uniform random policy.
        values: Current state-value estimate, shape (STATE_COUNT,).
    """

    def __init__(self, mdp: "Any", bellman_tolerance: float = 0.01) -> None:
        super().__init__(mdp, bellman_tolerance)
        self.policy = self._init_policy()

    def find_optimal_policy(self, max_iter: int = 99) -> np.ndarray:
        """
        Run policy iteration until the policy converges or max_iter is reached.

        Returns:
            The converged (optimal) policy array of shape (STATE_COUNT, ACTION_COUNT).
        """
        previous_policy: np.ndarray | None = None
        iteration = 0

        while not self._is_converged(previous_policy) and iteration < max_iter:
            self.evaluate_policy()
            previous_policy = np.copy(self.policy)
            self.improve_policy()
            iteration += 1

        return self.policy

    def evaluate_policy(self) -> None:
        """
        Estimate state values for the current policy via iterative Bellman updates.

        Applies the Bellman expectation equation repeatedly:
            V(s) = Σ_a π(a|s) Σ_s' T(s,a,s') [R(s,a,s') + V(s')]
        until the change in values falls below bellman_tolerance.

        Updates self.values in place.
        """
        value_history: list[np.ndarray] = []

        while self._bellman_error(value_history) > self.bellman_tolerance:
            # Expected next-state value for each (s, a): shape (S, A)
            expected_next_value = np.dot(self.mdp.state_action_transition_matrix, self.values)

            # Expected immediate reward for each (s, a): shape (S, A)
            expected_reward = (self.mdp.reward_matrix * self.mdp.state_action_transition_matrix).sum(axis=-1)

            # Weight by policy probabilities and collapse over actions: shape (S,)
            new_values = (self.policy * (expected_next_value + expected_reward)).sum(axis=-1)

            value_history.append(new_values)
            self.values = new_values

    def _init_policy(self) -> np.ndarray:
        """Return a uniform random policy of shape (STATE_COUNT, ACTION_COUNT)."""
        return np.full(
            (self.mdp.observation_space.n, self.mdp.action_space.n),
            1.0 / self.mdp.action_space.n,
        )

    def _is_converged(self, previous_policy: np.ndarray | None) -> bool:
        """Return True if the policy is unchanged since the last improvement step."""
        return previous_policy is not None and np.array_equal(previous_policy, self.policy)
