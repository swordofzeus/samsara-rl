from typing import Any

import numpy as np


class Memory:
    def __init__(self, observation_space: int, action_space: int, memory_size: int = 1000) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

        self.states: np.ndarray = (
            np.zeros((memory_size, observation_space))
            if observation_space > 2
            else np.zeros(
                memory_size,
            )
        )
        self.actions: np.ndarray = (
            np.zeros((memory_size, action_space))
            if action_space > 2
            else np.zeros(
                memory_size,
            )
        )
        self.rewards: np.ndarray = np.zeros(memory_size)
        self.memory_size = memory_size
        self.curr_index = 0

    def current_state(self) -> Any:
        return self.past_states()[-1]

    def past_states(self) -> np.ndarray:
        return self.states[0 : self.curr_index + 1]

    def past_actions(self) -> np.ndarray:
        return self.actions[0 : self.curr_index + 1]

    def past_rewards(self) -> np.ndarray:
        return self.rewards[0 : self.curr_index + 1]
