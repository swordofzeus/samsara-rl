from typing import Any

import numpy as np
import torch

from samsara_rl.control.function_approximation.functions.manual.linear import Node


class TDOptimizer:
    def __init__(self, q: Any, alpha: float, gamma: float, _lambda: float, eligibility_traces: list[np.ndarray]) -> None:
        self.q = q
        self.alpha = alpha
        self.gamma = gamma
        self._lambda = _lambda
        self.eligibility_traces = eligibility_traces

    def step(self, A: int, td_error: Any) -> None:
        with torch.no_grad():
            for index, param in enumerate(self.q.parameters()):
                curr_trace = self.eligibility_traces[index]
                curr_trace: Any = torch.from_numpy(curr_trace) if isinstance(param, torch.Tensor) else curr_trace  # type: ignore[no-redef]
                param.data += self.alpha * curr_trace * td_error
