import torch
from torch import nn


class DoubleDQNTarget:
    """Double DQN target computation.

    Decouples action selection from evaluation to reduce
    maximization bias. The online Q network selects the best
    action, and the target network evaluates it.
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
            q_network: The online Q network, used to select actions.
            target_network: The frozen target network, used to evaluate.
            S_prime: Next states, shape ``(batch, state_dim)``.
            terminal: Boolean tensor, shape ``(batch,)``.

        Returns:
            Target Q values, shape ``(batch,)``.
        """
        selected_next_action = q_network(S_prime).argmax(axis=1).unsqueeze(1).to(torch.long)
        Q_S_prime = torch.gather(target_network(S_prime), dim=1, index=selected_next_action)
        return torch.where(
            terminal.unsqueeze(1),
            0,
            Q_S_prime,
        ).squeeze(1)
