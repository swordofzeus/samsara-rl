from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from samsara_rl.agent import Agent
from samsara_rl.utils.bellman import discounted_cum_trajectory
from samsara_rl.utils.memory.episode import Episode


class MonteCarloPolicyGradient(Agent):
    """Monte Carlo Policy Gradient for episodic environments.

    Uses a neural network to parameterize the policy directly.
    Gradients are accumulated over a batch of episodes before
    updating the policy network.

    Args:
        mdp: Gymnasium-compatible environment.
        alpha: Learning rate.
        gamma: Discount factor.
        policy_network: Neural network that maps states to action logits.
        optimizer: Optimizer for the policy network. Defaults to Adam.
        batch_size: Number of episodes to accumulate gradients over
            before performing an optimizer step.
        use_advantage: If True, subtract a running average baseline
            from discounted returns to reduce variance.
    """

    def __init__(
        self,
        mdp: Any,
        alpha: float,
        gamma: float,
        policy_network: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        batch_size: int = 1,
        use_advantage: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(mdp, alpha, gamma, **kwargs)

        self.policy_network = policy_network
        self.optimizer = optimizer or torch.optim.Adam(self.policy_network.parameters(), lr=self.alpha)
        self.batch_size = batch_size
        self.running_average = 0
        self.episode_count = 0
        self.use_advantage = use_advantage
        self.metrics: dict[str, float] = {}

    def post_visit(self, history: Episode, terminal: bool) -> None:
        pass

    def advantage(self, rewards: np.ndarray) -> np.ndarray:
        """Compute advantage by subtracting a running average baseline.

        Updates the running average of episode returns using an
        incremental mean, then subtracts it from the discounted
        rewards to center them around zero.

        Args:
            rewards: Discounted returns for each time step.

        Returns:
            Advantage-adjusted rewards.
        """

        normalized_rewards = rewards - self.running_average
        self.running_average = self.running_average + (rewards[0] - self.running_average) / self.episode_count
        return normalized_rewards

    def train(self, history: Episode) -> float:
        """Compute the policy gradient loss and accumulate gradients.

        Runs the episode non terminal states (1...N-1) through
        the policy network, computes log-probabilities of the taken
        actions, and weights them by discounted returns.
        Calls ``backward()`` to accumulate gradients without stepping the optimizer.

        Args:
            history: The complete episode history.

        Returns:
            The scalar policy gradient loss for this episode.
        """
        self.episode_count += 1

        states = history.past_states()[:-1]
        rewards = history.past_rewards()[:-1]
        actions = history.past_actions()[:-1]

        discounted_rewards = discounted_cum_trajectory(self.gamma, rewards)
        discounted_rewards = self.advantage(discounted_rewards) if self.use_advantage else discounted_rewards
        discounted_rewards_tensor = torch.from_numpy(discounted_rewards.copy())

        action_values = self.policy_network(states)
        action_score = F.log_softmax(action_values, dim=1)
        selected_actions = torch.from_numpy(actions).to(torch.long).unsqueeze(1)
        selected_action_values = torch.gather(action_score, dim=1, index=selected_actions).squeeze(1)

        loss = (-1 / self.batch_size) * (discounted_rewards_tensor * selected_action_values).sum()
        loss.backward()
        return loss.item()

    def select_action(self, state: Any) -> int:
        """Sample an action from the current policy.

        Args:
            state: The current environment state.

        Returns:
            The selected action index.
        """
        with torch.no_grad():
            action_values = self.policy_network(state)
            action_score = F.softmax(action_values, dim=0)
            selected_action = torch.multinomial(action_score, num_samples=1, replacement=True)
            return int(selected_action.squeeze().item())

    def get_metrics(self, trajectory: Episode) -> dict[str, float]:
        return {**super().get_metrics(trajectory), **self.metrics}

    def post_episode(self, history: Episode) -> None:
        """Train on the episode and step the optimizer every batch.

        Args:
            history: The complete episode history.
        """
        loss = self.train(history)
        self.metrics["loss"] = loss
        if self.episode_count % self.batch_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
