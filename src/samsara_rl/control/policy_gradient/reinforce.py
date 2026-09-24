from typing import Any

import torch
from torch import nn

from samsara_rl.control.policy_gradient.monte_carlo_policy_gradient import (
    MonteCarloPolicyGradient,
)


class Reinforce(MonteCarloPolicyGradient):
    """REINFORCE algorithm (Williams, 1992).

    A special case of Monte Carlo Policy Gradient where the optimizer
    steps after every episode (batch_size=1).

    Args:
        mdp: Gymnasium-compatible environment.
        alpha: Learning rate.
        gamma: Discount factor.
        policy_network: Neural network that maps states to action logits.
        optimizer: Optimizer for the policy network. Defaults to Adam.
        use_advantage: If True, subtract a running average baseline
            from discounted returns to reduce variance.
    """

    def __init__(
        self,
        mdp: Any,
        alpha: float,
        gamma: float,
        policy_network: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        use_advantage: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mdp,
            alpha,
            gamma,
            policy_network,
            optimizer,
            batch_size=1,
            use_advantage=use_advantage,
            **kwargs,
        )
