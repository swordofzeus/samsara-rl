"""SARSA control with function approximation."""

from typing import Any

from samsara_rl.control.function_approximation.td import TemporalDifferenceGradient
from samsara_rl.utils.target import sarsa_target


class SarsaGradient(TemporalDifferenceGradient):
    """On-policy SARSA control using semi-gradient TD(lambda).

    Inherits the full semi-gradient TD(lambda) update from
    ``TemporalDifferenceGradient`` and fixes the TD target to
    Q(S', A'), where A' is the action actually taken under the
    current policy.

    Args:
        **kwargs: Forwarded to ``TemporalDifferenceGradient.__init__``.
            Required keys include ``mdp``, ``policy``, and ``q``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs, target=sarsa_target)
