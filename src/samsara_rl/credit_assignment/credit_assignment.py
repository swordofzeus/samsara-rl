from abc import ABC, abstractmethod

from samsara_rl.utils.memory.episode import Episode, Transition


class CreditAssignment(ABC):
    """Base class for credit assignment methods.

    Encapsulates the update rule that adjusts value estimates
    based on observed transitions. Subclasses implement ``observe``
    and ``terminal`` for their specific algorithm (e.g. TD, Monte Carlo).

    Args:
        alpha: Learning rate for incremental updates.
        gamma: Discount factor applied to future rewards.
    """

    def __init__(self, alpha: float = 0.01, gamma: float = 0.9) -> None:
        self.alpha = alpha
        self.gamma = gamma

    @abstractmethod
    def observe(self, transition: Transition, target: float | None = None) -> None:
        """Process a single transition (S, A, R, S').

        Called after each step. TD methods perform their update here.
        Monte Carlo methods may no-op.

        Args:
            transition: The most recent (S, A, R, S') transition.
            target: Optional scalar TD target. Used by TD methods,
                ignored by Monte Carlo.
        """
        pass

    @abstractmethod
    def terminal(self, history: Episode) -> None:
        """Called at the end of an episode.

        Monte Carlo methods perform their full update here.
        TD methods reset per-episode state (e.g. eligibility traces).
        """
        pass
