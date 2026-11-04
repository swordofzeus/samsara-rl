import torch
import torch.nn as nn


class LinearNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool) -> None:
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=bias, dtype=torch.float64)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)  # type: ignore[no-any-return]
