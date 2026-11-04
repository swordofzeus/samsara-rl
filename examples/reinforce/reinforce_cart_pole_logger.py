import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.categorical import Categorical

if TYPE_CHECKING:
    from samsara_rl.control.policy_gradient.monte_carlo_policy_gradient import MonteCarloPolicyGradient


# Probe states: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
ANGULAR_VELOCITIES = [-0.5, -0.25, 0, 0.25, 0.5]

PROBE_LOW = torch.tensor(
    [[0, 0, angle, av] for angle in [-0.05, 0, 0.05] for av in ANGULAR_VELOCITIES], dtype=torch.float32
)

PROBE_MID = torch.tensor(
    [[0, 0, angle, av] for angle in [-0.15, -0.1, 0.1, 0.15] for av in ANGULAR_VELOCITIES], dtype=torch.float32
)

PROBE_HIGH = torch.tensor(
    [[0, 0, angle, av] for angle in [-0.3, -0.2, 0.2, 0.3] for av in ANGULAR_VELOCITIES], dtype=torch.float32
)


class ReinforceCartPoleLogger:
    def __init__(
        self,
        probe_states: dict[str, torch.Tensor] | None = None,
        policy_surface_freq: int = 50,
        save_dir: str = "policy_surfaces",
    ) -> None:
        self.probe_states = probe_states or {
            "LowAngle_Safe": PROBE_LOW,
            "MidAngle_Moderate": PROBE_MID,
            "HighAngle_Danger": PROBE_HIGH,
        }
        self.policy_surface_freq = policy_surface_freq
        self.save_dir = save_dir
        self._visit_counts: dict[str, int] = {}
        self._total_visits: int = 0

    def on_visit(self, agent: "MonteCarloPolicyGradient") -> None:
        state = agent.mdp.unwrapped.state
        angle = abs(state[2])
        if angle < 0.05:
            region = "LowAngle_Safe"
        elif angle < 0.15:
            region = "MidAngle_Moderate"
        else:
            region = "HighAngle_Danger"
        self._visit_counts[region] = self._visit_counts.get(region, 0) + 1
        self._total_visits += 1

    def on_episode(self, agent: "MonteCarloPolicyGradient") -> None:
        if not agent.tensorboard:
            return

        self.episode = agent.curr_episode
        self.tb = agent.tensorboard

        if self.save_dir and not hasattr(self, "_save_dir_set"):
            self.save_dir = os.path.join(agent.tensorboard.log_dir, self.save_dir)
            self._save_dir_set = True

        self._log_policy_entropy(agent)
        self._log_action_split(agent)
        self._log_visitation(agent)
        self._log_loss(agent)
        self._log_policy_surface(agent)

    def _log_policy_entropy(self, agent: "MonteCarloPolicyGradient") -> None:
        with torch.no_grad():
            for name, probes in self.probe_states.items():
                logits = agent.policy_network(probes)
                entropy = Categorical(logits=logits).entropy().mean().item()
                self.tb.log_metric(epoch=self.episode, metric_name=f"PolicyEntropy/{name}", value=entropy)

    def _log_action_split(self, agent: "MonteCarloPolicyGradient") -> None:
        center = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        left_lean = torch.tensor([[0.0, 0.0, -0.1, 0.0]], dtype=torch.float32)
        right_lean = torch.tensor([[0.0, 0.0, 0.1, 0.0]], dtype=torch.float32)

        with torch.no_grad():
            for name, state in [("Center", center), ("LeftLean", left_lean), ("RightLean", right_lean)]:
                logits = agent.policy_network(state).squeeze()
                probs = Categorical(logits=logits).probs
                self.tb.log_metric(
                    epoch=self.episode, metric_name=f"Policy_Action/{name}_P_Left", value=probs[0].item()
                )
                self.tb.log_metric(
                    epoch=self.episode, metric_name=f"Policy_Action/{name}_P_Right", value=probs[1].item()
                )

    def _log_visitation(self, agent: "MonteCarloPolicyGradient") -> None:
        if self._total_visits == 0:
            return
        for region in ["LowAngle_Safe", "MidAngle_Moderate", "HighAngle_Danger"]:
            count = self._visit_counts.get(region, 0)
            pct = count / self._total_visits
            self.tb.log_metric(epoch=self.episode, metric_name=f"Visitation/{region}", value=pct)
        self._visit_counts = {}
        self._total_visits = 0

    def _log_loss(self, agent: "MonteCarloPolicyGradient") -> None:
        if hasattr(agent, "metrics") and "loss" in agent.metrics:
            self.tb.log_metric(epoch=self.episode, metric_name="Loss", value=agent.metrics["loss"])

    def _log_policy_surface(self, agent: "MonteCarloPolicyGradient") -> None:
        if self.episode % self.policy_surface_freq != 0:
            return

        angles = np.linspace(-0.4, 0.4, 30)
        ang_vels = np.linspace(-0.6, 0.6, 30)
        angle_grid, angvel_grid = np.meshgrid(angles, ang_vels)

        p_left = np.zeros_like(angle_grid)
        entropy = np.zeros_like(angle_grid)

        with torch.no_grad():
            for i in range(angle_grid.shape[0]):
                for j in range(angle_grid.shape[1]):
                    state = torch.tensor([0.0, 0.0, angle_grid[i, j], angvel_grid[i, j]], dtype=torch.float32)
                    logits = agent.policy_network(state)
                    dist = Categorical(logits=logits)
                    p_left[i, j] = dist.probs[0].item()
                    entropy[i, j] = dist.entropy().item()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={"projection": "3d"})

        axes[0].plot_surface(angle_grid, angvel_grid, p_left, cmap="coolwarm")
        axes[0].set_title(f"P(Push Left) — Episode {self.episode}")
        axes[0].set_xlabel("Pole Angle")
        axes[0].set_ylabel("Angular Velocity")
        axes[0].set_zlabel("P(Left)")

        axes[1].plot_surface(angle_grid, angvel_grid, entropy, cmap="viridis")
        axes[1].set_title(f"Policy Entropy — Episode {self.episode}")
        axes[1].set_xlabel("Pole Angle")
        axes[1].set_ylabel("Angular Velocity")
        axes[1].set_zlabel("Entropy")

        fig.tight_layout()
        self.tb.log_figure(epoch=self.episode, tag="PolicySurface", figure=fig)
        self.tb.flush()

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"policy_surface_ep{self.episode}.png")
            fig.savefig(save_path, dpi=100)
        plt.close(fig)
