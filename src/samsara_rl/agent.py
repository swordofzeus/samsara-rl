from abc import ABC, abstractmethod
from samsara_rl.search.sample_policy import SamplePolicy
from typing import Any

import numpy as np

from samsara_rl.search.search import Search



class Agent(ABC):
    """Base class for model-free policy evaluation methods.

    Provides the episode-generation loop (template method pattern) and
    defines hooks that subclasses override to implement algorithm-specific
    update logic.

    Args:
        mdp: Environment with ``STATE_COUNT``, ``ACTION_COUNT``,
            ``initial_state()``, ``is_terminal_state()``, and ``step()``
            methods.
        policy: Stochastic policy array of shape ``(S, A)`` where each
            row sums to 1.
        alpha: Learning rate for incremental Q updates.
        gamma: Discount factor applied to future rewards.
    """

    def __init__(self, mdp: Any, policy: np.ndarray, alpha: float = 0.01, gamma: float = 0.9, search:Search = None) -> None:
        self.q_table: np.ndarray = np.zeros((mdp.STATE_COUNT, mdp.ACTION_COUNT))
        self.mdp = mdp
        self.policy: np.ndarray = policy
        self.gamma: float = gamma
        self.alpha: float = alpha
        self.search = search if search else SamplePolicy()

    @abstractmethod
    def post_visit(self, trajectory: np.ndarray) -> None:
        """Called after each step within an episode.

        Subclasses use this to perform per-step updates (e.g. TD
        updates with eligibility traces). No-op for methods that
        only update at episode end (e.g. Monte Carlo).

        Args:
            trajectory: Array of shape ``(t, 3)`` containing all steps
                recorded so far in the current episode, where each row
                is ``[state, action, reward]``.
        """
        pass

    @abstractmethod
    def post_episode(self, trajectory: np.ndarray) -> None:
        """Called after a complete episode has been generated.

        Subclasses use this to perform end-of-episode updates
        (e.g. Monte Carlo return-based Q updates) or to reset
        per-episode state (e.g. eligibility traces).

        Args:
            trajectory: Array of shape ``(T, 3)`` containing the full
                episode, where each row is ``[state, action, reward]``.
        """
        pass

    def run_episode(self) -> np.ndarray:
        """Generate a complete episode under the current policy.

        Samples actions from the policy and steps through the MDP until
        a terminal state is reached. Calls ``post_visit`` after each
        step so subclasses can perform online updates.

        Returns:
            np.ndarray: Array of shape ``(T, 3)`` where each row is
                ``[state, action, reward]``.
        """
        curr_state, curr_step = self.mdp.initial_state(), 0

        trajectory = np.full((1000, 3), 0)

        while not self.mdp.is_terminal_state(curr_state):
            sampled_action = self.search.step(self.policy, curr_state, self.q_table)
            reward, next_state = self.mdp.step(curr_state, sampled_action)
            trajectory[curr_step][0] = curr_state
            trajectory[curr_step][1] = sampled_action
            trajectory[curr_step][2] = reward
            curr_step += 1
            curr_state = next_state
            self.post_visit(trajectory[0:curr_step])

        trajectory[curr_step][0] = curr_state

        self.post_visit(trajectory[0:curr_step+1])
        return trajectory[0:curr_step]

    def evaluate(self, max_iter: int = 1000) -> None:
        """Run prediction for a fixed number of episodes.

        Generates episodes and calls ``post_episode`` after each one,
        allowing subclasses to perform end-of-episode updates.

        Args:
            max_iter: Number of episodes to sample and learn from.
        """
        for _ in range(0, max_iter):
            trajectory = self.run_episode()
            self.post_episode(trajectory)
