from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from samsara_rl.utils.gym_utils import action_output_dim, state_output_dim
from samsara_rl.utils.logging.tensor_board import TensorBoardLogger
from samsara_rl.utils.memory.episode import Episode


class Agent(ABC):
    """Base class for model-free reinforcement learning agents.

    Provides the episode-generation loop (template method pattern) and
    defines hooks that subclasses override to implement algorithm-specific
    update logic.

    Args:
        mdp: Gymnasium-compatible environment.
        alpha: Learning rate for incremental Q updates.
        gamma: Discount factor applied to future rewards.
    """

    def __init__(
        self,
        mdp: Any,
        alpha: float = 0.01,
        gamma: float = 0.9,
        experiment_name: str | None = None,
        log_dir: str | None = None,
        autograd: bool = False,
        post_episode_hooks: list[Any] | None = None,
        post_visit_hooks: list[Any] | None = None,
    ) -> None:
        self.mdp = mdp
        self.gamma: float = gamma
        self.alpha: float = alpha
        self.rewards_across_episodes: list[float] = []
        self.curr_episode = 0
        self.tensorboard = (
            TensorBoardLogger(log_dir=log_dir, experiment_name=experiment_name)
            if (log_dir and experiment_name)
            else None
        )
        self.post_episode_hooks = post_episode_hooks if post_episode_hooks is not None else []
        self.post_visit_hooks = post_visit_hooks if post_visit_hooks is not None else []
        self.action_space = action_output_dim(self.mdp.action_space)
        self.observation_space = state_output_dim(self.mdp.observation_space)

    @abstractmethod
    def post_visit(self, history: Episode, terminal: bool) -> None:
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
    def post_episode(self, history: Episode) -> None:
        """Called after a complete episode has been generated.

        Subclasses use this to perform end-of-episode updates
        (e.g. Monte Carlo return-based Q updates) or to reset
        per-episode state (e.g. eligibility traces).

        Args:
            history: The complete episode history.
        """
        pass

    def register(self, hook: Any) -> None:
        if hasattr(hook, "on_episode"):
            self.post_episode_hooks.append(hook.on_episode)
        if hasattr(hook, "on_visit"):
            self.post_visit_hooks.append(hook.on_visit)

    @abstractmethod
    def select_action(self, state: Any) -> int:
        pass

    def run_episode(self) -> Episode:
        """Generate a complete episode under the current policy.

        Samples actions from the policy and steps through the MDP until
        a terminal state is reached. Calls ``post_visit`` after each
        step so subclasses can perform online updates.

        Returns:
            The episode history.
        """
        curr_state, _ = self.mdp.reset()
        episode_history = Episode.from_gym(self.mdp, curr_state)
        terminated = False

        while not terminated:
            curr_action = self.select_action(curr_state)
            next_state, reward, terminated, truncated, _ = self.mdp.step(curr_action)
            episode_history.record(curr_action, reward, next_state)
            curr_state = next_state

            terminated = terminated or truncated
            self.post_visit(episode_history, terminated)
            [hook(self) for hook in self.post_visit_hooks]

        return episode_history

    def evaluate(self, max_iter: int = 1000) -> None:
        """Run prediction for a fixed number of episodes.

        Generates episodes and calls ``post_episode`` after each one,
        allowing subclasses to perform end-of-episode updates.

        Args:
            max_iter: Number of episodes to sample and learn from.
        """
        for episode in range(0, max_iter):
            trajectory = self.run_episode()
            self.curr_episode += 1
            self.log_metrics(trajectory, episode)
            [hook(self) for hook in self.post_episode_hooks]
            self.post_episode(trajectory)
            if episode % 20 == 0 and self.tensorboard:
                self.tensorboard.flush()

    def get_metrics(self, trajectory: Episode) -> dict[str, float]:
        """Return metrics to log. Subclasses call super().get_metrics() and add their own."""
        return {"Reward": float(np.sum(trajectory.past_rewards()))}

    def log_metrics(self, trajectory: Episode, episode_number: int) -> None:
        if self.tensorboard:
            for name, value in self.get_metrics(trajectory).items():
                self.tensorboard.log_metric(
                    epoch=episode_number,
                    metric_name=name,
                    value=value,
                )
