from collections.abc import Callable
from enum import Enum

import numpy as np
import torch

from samsara_rl.policy.policy import Policy


class DecaySchedule(Enum):
    EXPONENTIAL = 1
    LINEAR = 2


class EpsilonGreedy(Policy):
    def __init__(
        self,
        q_values: Callable[..., np.ndarray],
        epsilon: float,
        epsilon_decay: float,
        epsilon_min: float = 0.05,
        epsilon_episodes: int | None = None,
    ) -> None:
        if epsilon > 1:
            raise ValueError(epsilon)
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_episodes = epsilon_episodes
        self.q_values = q_values
        self._episode_count = 0

    def decay(self, method: DecaySchedule = DecaySchedule.EXPONENTIAL) -> None:
        match method:
            case DecaySchedule.LINEAR:
                self._episode_count += 1
                self.epsilon = max(self.epsilon_min, self.epsilon_start - self._episode_count / self.epsilon_episodes)  # type: ignore[operator]
            case DecaySchedule.EXPONENTIAL:
                self.epsilon = (
                    self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
                )
            case _:
                raise ValueError(f"Unsupported decay schedule: {method}")

    def __call__(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
        q_output = self.q_values(state)
        state_action_values = torch.from_numpy(q_output) if isinstance(q_output, np.ndarray) else q_output
        batch_size = state.shape[0]
        action_count = state_action_values.shape[-1]

        random_action_values = torch.randint(action_count, (batch_size,), device=state_action_values.device)
        uniform_random = torch.rand(batch_size, device=state_action_values.device)

        selected_actions = torch.where(
            uniform_random < self.epsilon,
            random_action_values,
            state_action_values.argmax(dim=-1),
        )
        return selected_actions
