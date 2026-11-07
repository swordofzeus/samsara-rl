from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import gymnasium
import torch

from samsara_rl.functions.neural_networks.world_model_interpreter import (
    WorldModelInterpreter,
)


class Search(ABC):
    """Base class for forward search planning algorithms.

    Uses a learned world model and a rollout policy to simulate
    trajectories from a given state, evaluating candidate actions
    by their estimated discounted return.

    Subclasses implement the specific search strategy (e.g. Monte Carlo
    rollouts, tree search) in the ``plan`` method.

    Args:
        policy: Callable mapping state -> action, used during rollouts
            to select actions beyond the initial candidate.
        model: Environment model (e.g. a learned neural network) that
            predicts (next_state, reward, terminal) given (state, action).
        world_model_interpreter: Interpreter that parses the raw model
            output into next-state logits, reward, and terminal signal.
    """

    def __init__(
        self,
        policy: Callable[..., int],
        model: Any,
        world_model_interpreter: WorldModelInterpreter,
    ) -> None:
        self.policy = policy
        self.model = model
        self.world_model_interpreter = world_model_interpreter

    @abstractmethod
    def plan(
        self,
        state: Any,
        actions: gymnasium.spaces.Discrete,
        depth: int,
        budget: int,
        gamma: float = 1,
    ) -> torch.Tensor:
        """Search forward from state and return estimated values per action.

        Args:
            state: The current state to plan from.
            actions: The discrete action space to search over.
            depth: Maximum simulation depth per rollout.
            budget: Number of rollouts or simulations to run per action.
            gamma: Discount factor for future rewards (0 < gamma <= 1).

        Returns:
            Tensor of estimated values, one per action.

        Raises:
            ValueError: If gamma is not in (0, 1].
        """
        if gamma <= 0 or gamma > 1:
            raise ValueError("Gamma must be between 0-1")
        return torch.zeros(0)
