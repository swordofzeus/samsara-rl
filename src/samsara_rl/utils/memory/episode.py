from __future__ import annotations

from typing import Any

import numpy as np

from samsara_rl.utils.gym_utils import action_output_dim, state_output_dim
from samsara_rl.utils.memory.memory import Memory


class Episode(Memory):
    def __init__(self, observation_space: int, action_space: int, initial_state: Any) -> None:
        super().__init__(observation_space, action_space)
        self.states[0] = initial_state

    @classmethod
    def from_gym(cls, env: Any, s: Any) -> Episode:
        action_space = action_output_dim(env.action_space)
        state_space = state_output_dim(env.observation_space)
        return Episode(state_space, action_space, s)

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> Episode:
        history = Episode(1, 1, 1)
        history.states = states
        history.actions = actions
        history.rewards = rewards
        history.curr_index = len(states) - 1
        return history

    def record(self, action: Any, reward: float, s_prime: Any, a_prime: Any = None) -> None:
        self.states[self.curr_index + 1] = s_prime
        self.actions[self.curr_index] = action
        self.rewards[self.curr_index] = reward
        if a_prime:
            self.actions[self.curr_index + 1] = a_prime
        self.curr_index += 1
