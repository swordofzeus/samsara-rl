"""Q-Learning control with function approximation."""

from typing import Any

from samsara_rl.control.semi_gradient.semi_gradient_td_agent import SemiGradientTDAgent
from samsara_rl.utils.memory.episode import Episode


class QLearningGradient(SemiGradientTDAgent):
    """Off-policy Q-Learning control using semi-gradient TD(lambda).

    Uses max_a Q(S', a) as the TD target.
    """

    def __init__(self, mdp: Any, **kwargs: Any) -> None:
        super().__init__(mdp, **kwargs)

    def td_target(self, history: Episode, terminal: bool) -> float:
        """Compute TD target: 0 for terminal states, max_a Q(s', a) otherwise."""
        target = history.past_states()[-1]
        return 0 if terminal else self.q(target).max().item()
