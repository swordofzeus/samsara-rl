from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from samsara_rl.search.sample_policy import SamplePolicy
from samsara_rl.search.search import Search
from samsara_rl.utils.history import History


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

    def __init__(
        self,
        mdp: Any,
        policy: np.ndarray,
        alpha: float = 0.01,
        gamma: float = 0.9,
        search: Search | None = None,
    ) -> None:
        self.mdp = mdp
        self.policy: np.ndarray = policy
        self.gamma: float = gamma
        self.alpha: float = alpha
        self.search = search if search else SamplePolicy()
        self.rewards_across_episodes: list[float] = []
        self.curr_episode = 0

    @abstractmethod
    def post_visit(self, history: History, terminal: bool) -> None:
        """Called after each step within an episode.

        Subclasses use this to perform per-step updates (e.g. TD
        updates with eligibility traces). No-op for methods that
        only update at episode end (e.g. Monte Carlo).

        Args:
            history: The episode history recorded so far.
            terminal: Whether the episode has terminated.
        """
        pass

    @abstractmethod
    def post_episode(self, history: History) -> None:
        """Called after a complete episode has been generated.

        Subclasses use this to perform end-of-episode updates
        (e.g. Monte Carlo return-based Q updates) or to reset
        per-episode state (e.g. eligibility traces).

        Args:
            history: The complete episode history.
        """
        pass

    @abstractmethod
    def get_q_values(self, state: Any) -> np.ndarray:
        pass

    def run_episode(self) -> History:
        """Generate a complete episode under the current policy.

        Samples actions from the policy and steps through the MDP until
        a terminal state is reached. Calls ``post_visit`` after each
        step so subclasses can perform online updates.

        Returns:
            The episode history.
        """
        curr_state, _ = self.mdp.reset()
        episode_history = History.from_gym(self.mdp, curr_state)
        curr_action = self.search.step(self.policy, curr_state, self.get_q_values(episode_history.current_state()), 0)

        terminated = False

        while not terminated:
            next_state, reward, terminated, truncated, _ = self.mdp.step(curr_action)

            next_action = self.search.step(
                self.policy,
                next_state,
                self.get_q_values(next_state),
                self.curr_episode,
            )

            episode_history.record(curr_action, reward, next_state, next_action)
            curr_state = next_state
            curr_action = next_action

            terminated = terminated or truncated
            self.post_visit(episode_history, terminated)

        return episode_history

    def evaluate(self, max_iter: int = 1000) -> None:
        """Run prediction for a fixed number of episodes.

        Generates episodes and calls ``post_episode`` after each one,
        allowing subclasses to perform end-of-episode updates.

        Args:
            max_iter: Number of episodes to sample and learn from.
        """
        for _ in range(0, max_iter):
            trajectory = self.run_episode()
            self.curr_episode += 1
            self.post_episode(trajectory)
