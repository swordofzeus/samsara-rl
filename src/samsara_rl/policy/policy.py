from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    """Abstract base class for action selection policies."""

    @abstractmethod
    def __call__(self, state: Any) -> int:
        pass
