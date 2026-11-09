from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from samsara_rl.agent import Agent
from samsara_rl.functions.neural_networks.world_model_interpreter import (
    WorldModelInterpreter,
)
from samsara_rl.policy.policy import Policy
from samsara_rl.utils.gym_utils import action_output_dim, state_output_dim
from samsara_rl.utils.memory.episode import Episode
from samsara_rl.utils.memory.replay_buffer import ReplayBuffer


class WorldModelAgent(Agent):
    """Agent that collects experience and trains a world model.

    Runs episodes using a given policy, stores transitions in a replay
    buffer, and trains a world model to predict next states, rewards,
    and terminal signals.

    Args:
        mdp: Gymnasium-compatible environment.
        alpha: Learning rate for the world model optimizer.
        loss_fn: Loss function for next-state prediction.
        batch_size: Mini-batch size for world model training.
        world_model: Neural network that predicts (next_state, reward, terminal).
        world_model_interpreter: Interpreter for parsing world model outputs.
        optimizer: Optimizer for the world model. Defaults to Adam.
        policy: Policy used to collect experience.
        buffer_size: Maximum capacity of the replay buffer.
        save_path: File path to save the trained world model weights.
    """

    def __init__(
        self,
        mdp: Any,
        world_model: nn.Module,
        world_model_interpreter: WorldModelInterpreter,
        policy: Policy,
        loss_fn: Callable,
        alpha: float = 0.001,
        batch_size: int = 128,
        optimizer: torch.optim.Optimizer | None = None,
        buffer_size: int = 1000,
        save_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mdp,
            alpha,
            gamma=1,
            **kwargs,
        )
        self.world_model = world_model
        self.world_model_interpreter = world_model_interpreter
        self.replay_buffer = ReplayBuffer(
            state_output_dim(self.mdp.observation_space),
            action_output_dim(self.mdp.action_space),
            dtype=mdp.observation_space.dtype,
            memory_size=buffer_size,
        )
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(self.world_model.parameters(), lr=alpha)
        self.state_loss = loss_fn
        self.reward_loss = torch.nn.MSELoss()
        self.terminal_loss = torch.nn.BCEWithLogitsLoss()
        self.policy = policy
        self.batch_size = batch_size
        self.metrics: dict[str, float] = {}
        self.save_path = save_path

    def train(self) -> None:
        """Train the world model on a mini-batch from the replay buffer.

        Samples transitions, encodes actions as one-hot, and computes
        three losses: next-state, reward, and terminal prediction.
        """
        self.optimizer.zero_grad()
        S, A, R, S_prime, is_terminal = self.replay_buffer.sample(batch_size=self.batch_size)
        encoded_actions = F.one_hot(A.long(), num_classes=self.mdp.action_space.n).to(dtype=S.dtype, device=S.device)
        state_action = torch.hstack((S, encoded_actions))

        model_predictions = self.world_model(state_action)
        next_state_logits = self.world_model_interpreter.next_state_logits(model_predictions)
        predicted_reward = self.world_model_interpreter.predicted_reward(model_predictions)
        predicted_terminal = self.world_model_interpreter.predicted_terminal(model_predictions)
        state_target = self.world_model_interpreter.prepare_state_target(next_states=S_prime)

        next_state_loss = self.state_loss(
            next_state_logits,
            state_target,
        )
        reward_loss = self.reward_loss(predicted_reward, R.to(predicted_reward.dtype))
        terminal_loss = self.terminal_loss(predicted_terminal, is_terminal.float())
        total_loss = next_state_loss + reward_loss + terminal_loss
        total_loss.backward()
        self.optimizer.step()

        self.metrics.update({
            "World Model Loss": total_loss.item(),
            "State Loss": next_state_loss.item(),
            "Reward Loss": reward_loss.item(),
            "Terminal Loss": terminal_loss.item(),
        })

    def post_episode(self, history: Episode) -> None:
        """Called after each episode completes."""
        pass

    def post_visit(self, history: Episode, terminal: bool) -> None:
        """Record the latest transition and train if buffer is large enough.

        Args:
            history: The current episode history.
            terminal: Whether the episode has terminated.
        """
        if history.curr_index < 1:
            return
        self.replay_buffer.record(
            history.past_states()[-2],
            history.past_actions()[-2],
            history.past_rewards()[-2],
            history.past_states()[-1],
            terminal,
        )
        if self.replay_buffer.curr_index >= self.batch_size:
            self.train()

    def select_action(self, state: Any) -> int:
        """Select an action using the policy.

        Args:
            state: The current observation.

        Returns:
            The selected action index.
        """
        state_tensor = torch.as_tensor(state).unsqueeze(0)
        return int(self.policy(state_tensor)[0].item())

    def on_complete(self) -> None:
        if self.save_path and self.world_model is not None:
            torch.save(self.world_model.state_dict(), self.save_path)

    def get_metrics(self, trajectory: Episode) -> dict[str, float]:
        """Return training metrics including world model loss.

        Args:
            trajectory: The completed episode.

        Returns:
            Dictionary of metric names to values.
        """
        return {**super().get_metrics(trajectory), **self.metrics}
