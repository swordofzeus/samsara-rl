from abc import ABC, abstractmethod

import torch


class WorldModelInterpreter(ABC):
    """Interface for interpreting raw world model output.

    Subclasses define how to extract next state, reward, and terminal
    predictions from the world model's output tensor. This decouples
    Dyna-Q from environment-specific output formats.
    """

    @abstractmethod
    def predicted_next_state(self, output: torch.Tensor) -> torch.Tensor:
        """Extract the predicted next state from model output."""
        pass

    @abstractmethod
    def predicted_reward(self, output: torch.Tensor) -> torch.Tensor:
        """Extract the predicted reward from model output."""
        pass

    @abstractmethod
    def predicted_terminal(self, output: torch.Tensor) -> torch.Tensor:
        """Extract the predicted terminal probability from model output."""
        pass

    @abstractmethod
    def parse(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def next_state_logits(self, output: torch.Tensor) -> torch.Tensor:
        pass

    def is_terminal(self, terminal_logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return torch.where(torch.sigmoid(terminal_logits) > threshold, 1, 0)

    def prepare_state_target(self, next_states: torch.Tensor) -> torch.Tensor:
        return next_states
