from abc import ABC, abstractmethod

import numpy as np


class Search(ABC):
    """
    Select next action given policy, state, q table
    """

    @abstractmethod
    def step(self, policy: np.ndarray, state: int, q: np.ndarray) -> int:
        pass
