from typing import Any


class CreditAssignment:
    """Base class for credit assignment methods.

    Encapsulates the update rule that adjusts value estimates
    based on observed transitions. Subclasses implement ``update``
    for their specific algorithm (e.g. TD, Monte Carlo).

    Args:
        alpha: Learning rate for incremental updates.
        gamma: Discount factor applied to future rewards.
    """

    def __init__(self, alpha: float = 0.01, gamma: float = 0.9) -> None:
        self.alpha = alpha
        self.gamma = gamma

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update value estimates from a transition or trajectory."""
        pass

    def reset(self, **kwargs: Any) -> None:
        """Reset per-episode state (e.g. eligibility traces)."""
        pass
