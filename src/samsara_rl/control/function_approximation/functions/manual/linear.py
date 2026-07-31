from collections.abc import Callable
from typing import Any

import numpy as np


class Node:
    """Wraps a parameter array with its gradient, similar to a PyTorch Parameter.

    Attributes:
        value: The parameter tensor (numpy array).
        grad: Accumulated gradient, same shape as value.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape


    def __init__(self, value: np.ndarray) -> None:
        self.data: np.ndarray = value
        self.grad: np.ndarray = np.zeros(value.shape)

    def reset_grad(self) -> None:
        """Zero out the gradient."""
        self.grad = np.zeros(self.data.shape)


class LinearFunction:
    """Linear function approximator: Q(s) = X(s)^T W [+ B].

    Mimics a PyTorch-style interface with forward pass via __call__,
    gradient computation via backward, and parameter access via params.

    Args:
        feature_count: Number of input features (output dimension of X).
        action_count: Number of discrete actions.
        X: Feature extraction function mapping raw state to a feature
            vector of length feature_count. Defaults to identity.
        use_bias: Whether to include a bias term per action.

    """

    def __init__(
        self,
        feature_count: int,
        action_count: int,
        X: Callable[..., np.ndarray] | None = None,
        use_bias: bool = False,
    ) -> None:
        self.X: Callable[..., np.ndarray] = X if X else lambda x: x
        self.W: Node = Node(np.zeros((action_count, feature_count)))
        self.B: Node = Node(np.zeros(action_count))
        self.use_bias: bool = use_bias
        self.params: list[Node] = [self.W, self.B]
        self.context: list[tuple[Any, int]] = []

    def zero_grad(self) -> None:
        """Reset all parameter gradients and clear the forward pass context."""
        for param in self.params:
            param.reset_grad()
        self.context = []

    def __call__(self, S: Any, A: int | None = None) -> np.ndarray:
        """Forward pass. Records (S, A) in context if A is provided."""
        if A is not None:
            self.context.append((S, A))
        result: np.ndarray = self.W.data.dot(self.X(S))
        return result + self.B.data if self.use_bias else result

    def backward(self, upstream: float = 0) -> None:
        """Accumulate local gradients for all recorded (S, A) pairs."""
        for state, action in self.context:
            self.W.grad[action] += upstream * self.X(state)
            self.B.grad[action] += upstream * 1
        self.context = []

    def parameters(self) -> list["Node"]:
        return self.params
