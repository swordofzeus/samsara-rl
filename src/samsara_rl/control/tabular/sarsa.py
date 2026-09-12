from typing import Any

from samsara_rl.credit_assignment.temporal_difference import TemporalDifference
from samsara_rl.policy.epsilon_greedy import EpsilonGreedy
from samsara_rl.utils.memory.episode import Episode


class Sarsa(TemporalDifference):
    def __init__(self, mdp: Any, alpha: float = 0.01, gamma: float = 0.9):
        super().__init__(mdp, alpha, gamma)
        self.search = EpsilonGreedy(self.get_q_values, epsilon=0.99, epsilon_decay=0.98)

    def get_metrics(self, trajectory: Episode) -> dict[str, float]:
        metrics = super().get_metrics(trajectory)
        metrics["Epsilon"] = self.search.epsilon
        return metrics

    def select_action(self, state: Any) -> int:
        return self.search.step(state)

    def td_target(self, history: Episode) -> float:
        state = history.past_states()[-1].astype(int)
        A_prime = history.past_actions()[-1].astype(int)
        result: float = self.q[state][A_prime]
        return result
