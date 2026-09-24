from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        preprocess: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim, dtype=torch.float64)
        self.preprocess = preprocess

    def forward(self, x: Any) -> torch.Tensor:
        out: torch.Tensor = self.preprocess(x) if self.preprocess else x
        if isinstance(out, np.ndarray):
            out = torch.from_numpy(out).to(torch.float64)
        out = self.fc1(out)
        out = self.relu(out)
        return self.fc2(out)  # type: ignore[no-any-return]
