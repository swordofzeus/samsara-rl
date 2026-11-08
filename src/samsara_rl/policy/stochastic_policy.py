from typing import Any

import numpy as np
import torch

from samsara_rl.policy.policy import Policy


class StochasticPolicy(Policy):
    def __init__(self, distribution: np.ndarray) -> None:
        self.distribution = distribution

    def action_distribution(self, state: int) -> np.ndarray:
        return np.asarray(self.distribution[state])

    def __call__(self, state: Any) -> torch.Tensor:
        state_tensor = torch.as_tensor(state) if isinstance(state, np.ndarray) else state
        action_probability = torch.from_numpy(self.distribution)[state_tensor]
        sample = torch.multinomial(action_probability, num_samples=1, replacement=True)
        return sample.squeeze(-1)
