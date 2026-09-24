import struct
from samsara_rl.agent import Agent
from samsara_rl.utils.memory.episode import Episode
import torch
from torch.distributions.categorical import Categorical
import structlog
from typing import Any

logger = structlog.get_logger()

class ValueFunctionActorCritic(Agent):
    def __init__(
        self,
        mdp: Any,
        alpha: float = 0.01,
        gamma: float = 0.9,
        experiment_name: str | None = None,
        log_dir: str | None = None,
        autograd: bool = False,
        post_episode_hooks: list[Any] | None = None,
        post_visit_hooks: list[Any] | None = None,
        batch_size=32,
        policy_network=None,
        value_network=None,
    ) -> None:
        super().__init__(
            mdp,
            alpha,
            gamma,
            experiment_name,
            log_dir,
            autograd,
            post_episode_hooks,
            post_visit_hooks,
        )
        self.policy_network = policy_network
        self.value_network = value_network

    def post_episode(self, history: Episode) -> None:
        return

    def post_visit(self, history: Episode, terminal: bool) -> None:
        # critic network
        # td error
        # update value
        # policy network
        # backprop policy
        return

    def select_action(self, state: Any) -> int:
        with torch.no_grad():
            logits = self.policy_network(state)
            action = Categorical(logits=logits).sample().item()
            logger.debug("select_action", action=action, logits=logits)
            return action
