from collections.abc import Callable
from typing import Any

import structlog
import torch
from torch import nn

from samsara_rl.control.dqn.q_network import DQNAgent
from samsara_rl.functions.neural_networks.world_model_interpreter import (
    WorldModelInterpreter,
)
from samsara_rl.utils.memory.episode import Episode
from samsara_rl.utils.memory.replay_buffer import ReplayBuffer

logger = structlog.get_logger()


class DeepDynaQ(DQNAgent):
    """DQN agent augmented with a learned world model for planning.

    Extends DQNAgent with Dyna-style planning: after each real transition,
    trains a world model on real experience, then generates simulated
    transitions through the world model to further update the Q-network.
    """

    def __init__(
        self,
        mdp: Any,
        world_model: nn.Module,
        world_model_interpreter: WorldModelInterpreter,
        alpha: float = 0.001,
        gamma: float = 1,
        q: nn.Module | None = None,
        target: Any | None = None,
        target_update_freq: int = 1000,
        count_planning_updates: bool = False,
        epsilon: float = 1,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
        epsilon_episodes: int | None = None,
        epsilon_schedule: str = "linear",
        loss_fn: Callable | None = None,
        state_loss: Callable | None = None,
        batch_size: int = 128,
        model_plan_steps: int = 5,
        sim_refill_rate: int = 7,
        model_train_steps: int = 20,
        wm_batch_size: int = 32,
        min_wm_train_steps: int = 2,
        wm_train_decay: float | None = None,
        wm_train_decay_freq: int = 60,
        warmup: int = 20,
        buffer_size: int = 1000,
        lr_step_size: int | None = None,
        lr_gamma: float = 0.5,
        preprocess: Callable[..., torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mdp,
            alpha,
            gamma,
            q,
            target,
            target_update_freq,
            epsilon,
            epsilon_decay,
            epsilon_min,
            epsilon_episodes,
            epsilon_schedule,
            loss_fn,
            batch_size,
            buffer_size,
            lr_step_size,
            lr_gamma,
            **kwargs,
        )
        self.world_model = world_model
        self.world_model_interpreter = world_model_interpreter
        self.model_plan_steps = model_plan_steps
        self.model_train_steps = float(model_train_steps)
        self.wm_batch_size = wm_batch_size
        self.min_wm_train_steps = min_wm_train_steps
        self.wm_train_decay = wm_train_decay
        self.wm_train_decay_freq = wm_train_decay_freq
        self.warmup = warmup
        self.sim_buffer = ReplayBuffer(self.observation_space, self.action_space)
        self.preprocess = preprocess
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.alpha)
        self.state_loss = state_loss if state_loss else torch.nn.MSELoss()
        self.reward_loss = torch.nn.MSELoss()
        self.terminal_loss = torch.nn.BCEWithLogitsLoss()
        self.avg_episode_length: float = 0
        self.sim_refill_rate = sim_refill_rate
        self.count_planning_updates = count_planning_updates
        self._last_wm_loss: float | None = None

    def post_episode(self, history: Episode) -> None:
        self.avg_episode_length = (
            self.avg_episode_length + (history.curr_index - self.avg_episode_length) / self.curr_episode
        )
        if self.wm_train_decay and self.curr_episode % self.wm_train_decay_freq == 0:
            self.model_train_steps = max(self.min_wm_train_steps, self.model_train_steps * self.wm_train_decay)
        return super().post_episode(history)

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """One-hot encode a batch of discrete action indices.

        Args:
            actions: Action indices of shape (batch_size,).

        Returns:
            One-hot tensor of shape (batch_size, num_actions).
        """
        one_hot = torch.zeros((actions.shape[0], self.mdp.action_space.n))
        one_hot.scatter_(1, actions.long().unsqueeze(1), value=1)
        return one_hot

    def train_world_model(self) -> torch.Tensor:
        """Train the world model on a mini-batch of real transitions.

        Samples from the replay buffer, feeds (state, action) into the
        world model, and computes three losses:
        - state_loss for next-state prediction (default MSELoss)
        - MSELoss for reward prediction (continuous scalar)
        - BCEWithLogitsLoss for terminal prediction (binary)

        Returns:
            The combined loss scalar.
        """
        self.world_model_optimizer.zero_grad()

        states, actions, rewards, next_states, is_terminal = self.replay_buffer.sample(self.wm_batch_size)
        if self.preprocess:
            states = self.preprocess(states)

        action_encoding = self.encode_actions(actions)
        state_action_input = torch.hstack([states, action_encoding])
        output = self.world_model(state_action_input)

        next_state_logits = self.world_model_interpreter.next_state_logits(output)
        predicted_reward = self.world_model_interpreter.predicted_reward(output)
        predicted_terminal = self.world_model_interpreter.predicted_terminal(output)

        next_state_loss = self.state_loss(
            next_state_logits,
            self.world_model_interpreter.prepare_state_target(next_states),
        )
        reward_loss = self.reward_loss(predicted_reward, rewards)
        terminal_loss = self.terminal_loss(predicted_terminal, is_terminal.float())

        total_loss = next_state_loss + reward_loss + terminal_loss

        total_loss.backward()
        self.world_model_optimizer.step()
        return total_loss  # type: ignore[no-any-return]

    def fill_sim_buffer(self) -> None:
        """Generate a simulated episode through the world model.

        Starts from a random state sampled from the replay buffer,
        selects actions via epsilon-greedy, steps through the world
        model for (next_state, reward, terminal), and stores each
        transition in the sim_buffer.
        """
        is_terminal: bool | torch.Tensor = False
        state = self.replay_buffer.sample(1)[0]
        step_count = 0

        while not is_terminal and step_count < self.avg_episode_length:
            one_hot_state = self.preprocess(state) if self.preprocess else state.clone()
            action = self.search.step(state.squeeze().numpy())
            one_hot_action = self.encode_actions(torch.tensor([action]))
            state_action_input = torch.cat((one_hot_state, one_hot_action), dim=1)

            model_output = self.world_model(state_action_input)
            _, reward, terminal_logits = self.world_model_interpreter.parse(model_output)
            next_state = self.world_model_interpreter.predicted_next_state(model_output)
            is_terminal = self.world_model_interpreter.is_terminal(terminal_logits).any()

            self.sim_buffer.record(
                state.squeeze(),
                action,
                reward.squeeze().item(),
                next_state.squeeze(),
                bool(is_terminal.squeeze()),
            )
            step_count += 1
            state = next_state

    def post_visit(self, history: Episode, terminal: bool) -> None:
        """Perform DQN update, then train world model and run planning.

        After the standard DQN post_visit update, trains the world model
        on real transitions. Once past the warmup period, generates
        simulated episodes through the world model and uses them to
        further update the Q-network.
        """
        super().post_visit(history, terminal)
        if self.model_plan_steps == 0 or self.replay_buffer.curr_index < self.batch_size:
            return

        for _ in range(int(self.model_train_steps)):
            world_model_loss = self.train_world_model()
        self._last_wm_loss = world_model_loss.item()

        if self.curr_episode > self.warmup:
            for _ in range(self.sim_refill_rate):
                with torch.no_grad():
                    self.fill_sim_buffer()

            if self.sim_buffer.curr_index < self.batch_size:
                return

            for _ in range(self.model_plan_steps):
                self.train(replay_buffer=self.sim_buffer)
                if self.count_planning_updates:
                    self.target_counter += 1
