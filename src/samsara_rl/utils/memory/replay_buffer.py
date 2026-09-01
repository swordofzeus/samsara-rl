from typing import Any

import numpy as np
import torch

from samsara_rl.utils.memory.memory import Memory


class ReplayBuffer(Memory):
    def __init__(self, observation_space: int, action_space: int) -> None:
        super().__init__(observation_space, action_space)
        self.state_prime = np.zeros(self.states.shape)
        self.is_terminal = np.zeros(self.memory_size, dtype=bool)
        self.random_generator = np.random.default_rng()

    def current(self) -> int:
        return self.curr_index % self.memory_size

    def record(self, s: Any, action: Any, reward: float, s_prime: Any, is_terminal: bool = False) -> None:
        self.states[self.current()] = s
        self.state_prime[self.current()] = s_prime
        self.rewards[self.current()] = reward
        self.actions[self.current()] = action
        self.is_terminal[self.current()] = is_terminal
        self.curr_index += 1

    def next(self, index: int) -> int:
        return index % self.memory_size

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        random_indices = self.random_generator.choice(
            min(self.curr_index, self.memory_size), size=batch_size, replace=False
        )
        return (
            torch.from_numpy(self.states[random_indices]),
            torch.tensor(self.actions[random_indices]),
            torch.tensor(self.rewards[random_indices]),
            torch.tensor(self.state_prime[random_indices]),
            torch.tensor(self.is_terminal[random_indices]),
        )
