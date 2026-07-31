from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn


class LinearNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, bias: bool, preprocess: Callable[..., torch.Tensor] | None = None) -> None:
        super(LinearNetwork, self).__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=bias, dtype=torch.float64)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias) if bias else None
        self.preprocess = preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.from_numpy(np.atleast_1d(x))
        x = self.preprocess(x) if self.preprocess else x
        return self.w(x)  # type: ignore[no-any-return]
