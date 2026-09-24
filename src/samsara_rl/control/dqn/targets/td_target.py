import torch
from torch import nn


class DQNTarget:
    """Standard DQN target computation.

    Uses the target network to both select and evaluate the
    best next action: max_a Q_target(S', a).
    """

    def __call__(
        self,
        q_network: nn.Module,
        target_network: nn.Module,
        S_prime: torch.Tensor,
        terminal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target Q values for a batch of next states.

        Args:
            q_network: The online Q network (unused in standard DQN).
            target_network: The frozen target network.
            S_prime: Next states, shape ``(batch, state_dim)``.
            terminal: Boolean tensor, shape ``(batch,)``.

        Returns:
            Target Q values, shape ``(batch,)``.
        """
        return torch.where(terminal.unsqueeze(1), 0, target_network(S_prime)).max(dim=1).values
