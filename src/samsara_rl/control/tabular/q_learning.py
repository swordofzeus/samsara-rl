from typing import Any

from samsara_rl.control.tabular.tabular_agent import TabularAgent
from samsara_rl.utils.memory.episode import Episode


class QLearning(TabularAgent):
    def __init__(self, mdp: Any, **kwargs: Any):
        super().__init__(mdp, **kwargs)

    def td_target(self, history: Episode) -> float:
        state = history.past_states()[-1].astype(int)
        result: float = self.q[state].max()
        return result
