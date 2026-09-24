from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from samsara_rl.utils.gym_utils import action_output_dim, state_output_dim
from samsara_rl.utils.memory.memory import Memory


@dataclass
class Transition:
    """A single (S, A, R, S') transition, with optional A' for SARSA."""

    state: Any
    action: int
    reward: float
    next_state: Any
    next_action: int | None = None


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

    def record(
        self,
        action: Any,
        reward: float,
        s_prime: Any,
        a_prime: Any = None,
    ) -> None:
        self.actions[self.curr_index] = action
        self.rewards[self.curr_index] = reward
        self.states[self.curr_index + 1] = s_prime
        self.curr_index += 1
        self.actions[self.curr_index] = (
            a_prime if a_prime is not None else self.actions[self.curr_index]
        )  # explicit check of None because 0 can be a valid action and return False

    def last_transition(self, include_next_action: bool = False) -> Transition | None:
        """Return the most recent (S, A, R, S') transition.

        Args:
            include_next_action: If True, populate next_action with A'
                from the current index (used by SARSA).

        Returns:
            The transition, or None if no transitions have been recorded.
        """
        if self.curr_index < 1:
            return None
        S = self.past_states()[-2]
        A = int(self.past_actions()[-2])
        R = float(self.past_rewards()[-2])
        S_prime = self.past_states()[-1]
        A_prime = int(self.past_actions()[-1]) if include_next_action else None
        return Transition(state=S, action=A, reward=R, next_state=S_prime, next_action=A_prime)
