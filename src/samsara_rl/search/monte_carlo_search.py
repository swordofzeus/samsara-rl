from typing import Any

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F

from samsara_rl.search.search import Search
from samsara_rl.utils.bellman import discounted_cum_trajectory


class MonteCarloSearch(Search):
    """Monte Carlo forward search using a learned world model.

    Evaluates each candidate action by performing multiple stochastic
    rollouts through the world model, averaging the discounted returns.
    After the initial candidate action, subsequent actions are chosen
    by the rollout policy.
    """

    def rollout(
        self,
        state: torch.Tensor,
        action: int,
        depth: int,
        action_count: int,
        gamma: float,
    ) -> float:
        """Simulate a single rollout and return its discounted return.

        Unrolls the world model for up to ``depth`` steps starting
        from the given state and action. Subsequent actions are
        selected by ``self.policy``. Stops early if the model
        predicts a terminal state.

        Args:
            state: Current state tensor.
            action: Initial action to take.
            depth: Maximum number of steps to simulate.
            action_count: Size of the discrete action space.
            gamma: Discount factor for future rewards.

        Returns:
            The discounted return (G_0) of this rollout.
        """
        rewards = np.zeros(depth)
        for step in range(0, depth):
            one_hot_action = F.one_hot(torch.tensor(action), num_classes=action_count)
            state_action = torch.hstack((state, one_hot_action)).float()
            prediction = self.model(state_action)
            _, predicted_reward, terminal_logits = self.world_model_interpreter.parse(prediction.unsqueeze(0))
            predicted_next_state = self.world_model_interpreter.predicted_next_state(prediction.unsqueeze(0)).squeeze()
            predicted_is_terminal = self.world_model_interpreter.is_terminal(terminal_logits, threshold=0.5)
            rewards[step] = predicted_reward
            if predicted_is_terminal:
                break
            action = self.policy(predicted_next_state)
            state = predicted_next_state

        return float(discounted_cum_trajectory(gamma, rewards)[0])

    def plan(
        self,
        state: Any,
        actions: gymnasium.spaces.Discrete,
        depth: int,
        budget: int,
        gamma: float = 1,
    ) -> torch.Tensor:
        """Evaluate all actions via Monte Carlo rollouts.

        For each action in the action space, runs ``budget`` rollouts
        of length ``depth`` and averages the discounted returns.

        Args:
            state: The current state to plan from.
            actions: The discrete action space to search over.
            depth: Maximum simulation depth per rollout.
            budget: Number of rollouts to average per action.
            gamma: Discount factor for future rewards (0 < gamma <= 1).

        Returns:
            Tensor of average estimated values, one per action.
        """
        super().plan(state, actions, depth, budget, gamma=gamma)
        num_actions: int = int(actions.n)
        rewards = torch.zeros(num_actions)
        with torch.no_grad():
            for action in range(num_actions):
                for _ in range(budget):
                    rewards[action] = rewards[action] + self.rollout(state, action, depth, num_actions, gamma)
                rewards[action] = rewards[action] / budget
        return rewards
