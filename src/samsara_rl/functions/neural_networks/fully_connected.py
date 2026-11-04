import torch
import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=dtype)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        return self.fc2(out)  # type: ignore[no-any-return]
