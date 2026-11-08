from abc import ABC, abstractmethod
from typing import Any

import torch


class Policy(ABC):
    """Abstract base class for action selection policies."""

    @abstractmethod
    def __call__(self, state: Any) -> torch.Tensor:
        pass
