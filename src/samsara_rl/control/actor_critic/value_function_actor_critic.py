from typing import Any

import structlog
import torch
from torch.distributions.categorical import Categorical

from samsara_rl.agent import Agent
from samsara_rl.utils.memory.episode import Episode

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
        batch_size: int = 32,
        policy_network: Any = None,
        policy_optimizer: Any = None,
        value_network: Any = None,
        value_optimizer: Any = None,
        normalize_advantage: bool = False,
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
        self.loss = torch.nn.MSELoss()
        self.value_optimizer = (
            torch.optim.Adam(self.value_network.parameters(), lr=0.01) if value_optimizer is None else value_optimizer
        )
        self.policy_optimizer = (
            torch.optim.Adam(self.policy_network.parameters(), lr=0.001)
            if policy_optimizer is None
            else policy_optimizer
        )
        self.curr_episode = 0
        self.batch_size = batch_size
        self.update_policy = True

    def post_episode(self, history: Episode) -> None:
        S = history.past_states()
        R = torch.from_numpy(history.past_rewards())
        A = torch.from_numpy(history.past_actions())
        V_S = self.value_network(S).squeeze()
        V_S_prime = self.value_network(S[1:]).squeeze()
        V_S_prime = torch.hstack((V_S_prime, torch.tensor([0]))).detach()
        td_target = R + self.gamma * V_S_prime
        loss = self.loss(V_S, td_target)
        loss.backward()

        if self.update_policy:
            advantage_estimate = td_target.detach() - V_S.detach()

            logits = self.policy_network(S)
            action_values = Categorical(logits=logits).log_prob(A)

            policy_loss = -(action_values * advantage_estimate).sum()
            policy_loss.backward()

        if self.curr_episode % self.batch_size == 0:
            self.value_optimizer.step()
            self.value_network.zero_grad()

            if self.update_policy:
                self.policy_optimizer.step()
                self.policy_network.zero_grad()
        return

    def post_visit(self, history: Episode, terminal: bool) -> None:
        return

    def get_metrics(self, trajectory: Episode) -> dict[str, float]:
        metrics = super().get_metrics(trajectory)
        self.last_reward = metrics["Reward"]
        return metrics

    def select_action(self, state: Any) -> int:
        with torch.no_grad():
            logits = self.policy_network(state)
            action: int = Categorical(logits=logits).sample().item()
            return action
