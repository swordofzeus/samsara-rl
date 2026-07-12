from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Search(ABC):
    """
    Select next action given policy, state, q table
    """

    @abstractmethod
    def step(self, policy: np.ndarray, state: Any, q: np.ndarray, episode: Any = None) -> int:
        pass
