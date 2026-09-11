from abc import abstractmethod
from typing import Any

import numpy as np
import torch

from samsara_rl.agent import Agent
from samsara_rl.utils.memory.episode import Episode


class TemporalDifference(Agent):
    def __init__(
        self,
        mdp: Any,
        alpha: float = 0.01,
        gamma: float = 0.9,
        _lambda: float = 0.4,
    ) -> None:
        super().__init__(mdp, alpha=alpha, gamma=gamma)
        self.eligibility = np.zeros(
            (self.mdp.observation_space.n, self.mdp.action_space.n)
        )
        self._lambda = _lambda
        self.q = np.zeros((mdp.observation_space.n, mdp.action_space.n))

    @abstractmethod
    def td_target(self, history: Episode) -> float:
        pass

    def get_q_values(self, state: int) -> np.ndarray:
        result: np.ndarray = self.q[state]
        return result

    def post_episode(self, history: Episode) -> None:
        self.eligibility = np.zeros(
            (self.mdp.observation_space.n, self.mdp.action_space.n)
        )

    def post_visit(self, history: Episode, terminal: bool = False) -> None:
        if history.curr_index < 1:
            return
        self.eligibility = self.eligibility * self._lambda
        visited_state = history.past_states()[-2].astype(int)
        el_action = history.past_actions()[-2].astype(int)
        self.eligibility[visited_state][el_action.astype(np.int8)] = 1

        R_prime = history.past_rewards()[-2]
        with torch.no_grad():
            Q_expectation = self.td_target(history)

        S = history.past_states()[-2].astype(int)
        A = history.past_actions()[-2].astype(np.int8)
        Q = self.q[S][A]

        td_error = (R_prime + self.gamma * Q_expectation) - Q
        self.q = self.q + self.alpha * self.eligibility * td_error
