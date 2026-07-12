"""Q-Learning control with function approximation."""

from samsara_rl.control.function_approximation.td import TemporalDifferenceGradient
from samsara_rl.utils.target import qlearning_target


class QLearningGradient(TemporalDifferenceGradient):
    """Off-policy Q-Learning control using semi-gradient TD(lambda).

    Inherits the full semi-gradient TD(lambda) update from
    ``TemporalDifferenceGradient`` and fixes the TD target to
    max_a Q(S', a), selecting the greedy action regardless of
    the policy used for exploration.

    Args:
        **kwargs: Forwarded to ``TemporalDifferenceGradient.__init__``.
            Required keys include ``mdp``, ``policy``, and ``q``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, target=qlearning_target, _lambda=1)
