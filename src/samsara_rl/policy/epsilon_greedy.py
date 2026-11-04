import random
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

    def step(self, state: np.ndarray | torch.Tensor) -> int:
        action_values = self.q_values(state)
        exploit = random.uniform(0, 1) > self.epsilon
        sampled_action = action_values.argmax() if exploit else random.randint(0, action_values.shape[0] - 1)
        return int(sampled_action)
