import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import structlog
import torch
from torch import nn

from samsara_rl.agent import Agent
from samsara_rl.control.function_approximation.batch.deep_q_network.targets.td_target import (
    DQNTarget,
)
from samsara_rl.search.epsilon_greedy import EpsilonGreedy
from samsara_rl.utils.memory.episode import Episode
from samsara_rl.utils.memory.replay_buffer import ReplayBuffer

logger = structlog.get_logger()

MIN_BUFFER_SIZE = 512


class QNetwork(Agent):
    """Deep Q-Network agent with experience replay and target network.

    Uses a neural network to approximate Q values and trains via
    mini-batch gradient descent on transitions sampled from a replay
    buffer. A frozen target network provides stable TD targets.

    Args:
        mdp: Gymnasium environment.
        policy: Stochastic policy array of shape ``(S, A)``.
        alpha: Learning rate for the Adam optimizer.
        gamma: Discount factor applied to future rewards.
        q: Neural network module that maps states to Q values.
        target: Target computation strategy. Defaults to ``DQNTarget``.
        target_update_freq: Steps between hard target network swaps.
        epsilon: Initial exploration rate for epsilon-greedy.
        epsilon_decay: Multiplicative decay applied to epsilon each episode.
        loss_fn: Loss function for training. Defaults to ``MSELoss``.
    """

    def __init__(
        self,
        mdp: Any,
        policy: np.ndarray,
        alpha: float = 0.001,
        gamma: float = 1,
        q: nn.Module | None = None,
        target: Any | None = None,
        target_update_freq: int = 1000,
        epsilon: float = 1,
        epsilon_decay: float = 0.999,
        loss_fn: Callable | None = None,
        batch_size: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mdp,
            policy,
            alpha,
            gamma,
            **kwargs,
        )
        self.q: nn.Module = q  # type: ignore[assignment]
        self.search = EpsilonGreedy(epsilon=epsilon, epsilon_decay=epsilon_decay)
        self.replay_buffer = ReplayBuffer(self.observation_space, self.action_space)
        self.target_network = copy.deepcopy(self.q)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.alpha)
        self.loss = loss_fn if loss_fn else torch.nn.MSELoss()
        self.target_update_freq = target_update_freq
        self.target_counter = 0
        self.td_target_range: list[torch.Tensor] = []
        self.target = target if target else DQNTarget()
        self.batch_size = batch_size

    def _build_up_replay_buffer(self, episode: Episode, terminal: bool) -> None:
        """Record the latest transition into the replay buffer.

        Args:
            episode: The current episode history.
            terminal: Whether the episode has terminated.
        """
        self.replay_buffer.record(
            episode.past_states()[-2],
            episode.past_actions()[-2],
            episode.past_rewards()[-2],
            episode.past_states()[-1],
            terminal,
        )

    def train(self) -> None:
        """Sample a mini-batch from the replay buffer and perform one gradient step."""
        self.optimizer.zero_grad()
        S, A, R, S_prime, terminal = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            Q_S_prime = self.target(self.q, self.target_network, S_prime, terminal)

        selected_actions = A.unsqueeze(1).to(torch.long)
        Q_S = torch.gather(self.q(S), dim=1, index=selected_actions).squeeze(1)
        td_target = R + self.gamma * Q_S_prime
        self.td_target_range = [torch.min(td_target), torch.max(td_target)]
        mse_loss = self.loss(Q_S, td_target)
        self.last_loss = mse_loss.item()
        mse_loss.backward()
        self.optimizer.step()

    def post_visit(self, history: Episode, terminal: bool) -> None:
        """Called after each step within an episode.

        Records the transition, trains on a mini-batch if the replay
        buffer is sufficiently full, and periodically swaps the target
        network.

        Args:
            history: The episode history recorded so far.
            terminal: Whether the episode has terminated.
        """
        if history.curr_index < 1:
            return

        self._build_up_replay_buffer(history, terminal)

        if self.replay_buffer.curr_index >= MIN_BUFFER_SIZE:
            self.target_counter += 1
            self.train()

        if self.target_counter >= self.target_update_freq:
            self.target_network = copy.deepcopy(self.q)
            self.target_counter = 0
            logger.debug("target_network_swap", episode=self.curr_episode)

    def post_episode(self, history: Episode) -> None:
        """Called after a complete episode. Decays the exploration rate.

        Args:
            history: The complete episode history.
        """
        self.search.decay()

    def get_q_values(self, state: Any) -> np.ndarray:
        """Return Q values for all actions at the given state.

        Args:
            state: The state to evaluate.

        Returns:
            Array of Q values, one per action.
        """
        with torch.no_grad():
            return np.array(self.q(state).numpy(), dtype=np.float64)
