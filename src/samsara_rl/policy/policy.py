from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Policy(ABC):
    """Abstract base class for action selection policies."""

    @abstractmethod
    def step(self, state: Any) -> int:
        pass
