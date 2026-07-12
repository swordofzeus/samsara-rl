from __future__ import annotations

from typing import Any

import numpy as np

from samsara_rl.utils.gym_utils import action_output_dim, state_output_dim


class History:
    def __init__(self, observation_space: int, action_space: int, initial_state: Any) -> None:
        self.states: np.ndarray = (
            np.zeros((1000, observation_space))
            if observation_space > 2
            else np.zeros(
                1000,
            )
        )
        self.actions: np.ndarray = (
            np.zeros((1000, action_space))
            if action_space > 2
            else np.zeros(
                1000,
            )
        )
        self.rewards: np.ndarray = np.zeros(1000)
        self.states[0] = initial_state
        self.curr_index: int = 0

    def current_state(self) -> Any:
        return self.past_states()[-1]

    def past_states(self) -> np.ndarray:
        return self.states[0 : self.curr_index + 1]

    def past_actions(self) -> np.ndarray:
        return self.actions[0 : self.curr_index + 1]

    def past_rewards(self) -> np.ndarray:
        return self.rewards[0 : self.curr_index + 1]

    def record(self, action: Any, reward: float, s_prime: Any, a_prime: Any = None) -> None:
        self.states[self.curr_index + 1] = s_prime
        self.actions[self.curr_index] = action
        self.rewards[self.curr_index] = reward
        if a_prime:
            self.actions[self.curr_index + 1] = a_prime
        self.curr_index += 1

    @classmethod
    def from_gym(cls, env: Any, s: Any) -> History:
        action_space = action_output_dim(env.action_space)
        state_space = state_output_dim(env.observation_space)
        return History(state_space, action_space, s)
