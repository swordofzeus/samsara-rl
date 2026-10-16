import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import structlog
import torch
from torch.distributions.categorical import Categorical

logger = structlog.get_logger()

if TYPE_CHECKING:
    from samsara_rl.agent import Agent


# Probe states: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
# Cart position and velocity fixed at 0, varying angle and angular velocity
ANGULAR_VELOCITIES = [-0.5, -0.25, 0, 0.25, 0.5]

PROBE_LOW = torch.tensor(
    [[0, 0, angle, av] for angle in [-0.05, 0, 0.05] for av in ANGULAR_VELOCITIES], dtype=torch.float64
)

PROBE_MID = torch.tensor(
    [[0, 0, angle, av] for angle in [-0.15, -0.1, 0.1, 0.15] for av in ANGULAR_VELOCITIES], dtype=torch.float64
)

PROBE_HIGH = torch.tensor(
    [[0, 0, angle, av] for angle in [-0.3, -0.2, 0.2, 0.3] for av in ANGULAR_VELOCITIES], dtype=torch.float64
)


class CartPoleLogger:
    def __init__(
        self,
        probe_states: dict[str, torch.Tensor] | None = None,
        decision_boundary_freq: int = 20,
        save_dir: str = "decision_surfaces",
        freeze_threshold: float = 450,
        freeze_window: int = 10,
    ) -> None:
        self.probe_states = probe_states or {
            "LowAngle_Safe": PROBE_LOW,
            "MidAngle_Moderate": PROBE_MID,
            "HighAngle_Danger": PROBE_HIGH,
        }
        self.decision_boundary_freq = decision_boundary_freq
        self.save_dir = save_dir
        self.freeze_threshold = freeze_threshold
        self.freeze_window = freeze_window
        self.reward_history: list[float] = []
        self._visit_counts: dict[str, int] = {}
        self._total_visits: int = 0

    def on_visit(self, agent: "Agent") -> None:
        if not hasattr(agent, "value_network"):
            return
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

    def on_episode(self, agent: "Agent") -> None:
        if not agent.tensorboard:
            return

        self.episode = agent.curr_episode
        self.tb = agent.tensorboard

        if self.save_dir and not hasattr(self, "_save_dir_set"):
            self.save_dir = os.path.join(agent.tensorboard.log_dir, self.save_dir)
            self._save_dir_set = True

        self._check_freeze_policy(agent)
        self._log_value_by_danger(agent)
        self._log_policy_entropy(agent)
        self._log_center_action_distribution(agent)
        self._log_advantage_by_action(agent)
        self._log_visitation(agent)
        self._log_value_surface(agent)

    def _check_freeze_policy(self, agent: "Agent") -> None:
        if not agent.update_policy:
            return
        if not hasattr(agent, "last_reward"):
            return
        self.reward_history.append(agent.last_reward)
        if len(self.reward_history) >= self.freeze_window:
            recent = self.reward_history[-self.freeze_window :]
            if all(r > self.freeze_threshold for r in recent):
                agent.update_policy = False
                logger.info("freezing policy", episode=self.episode)

    def _log_value_by_danger(self, agent: "Agent") -> None:
        with torch.no_grad():
            for name, probes in self.probe_states.items():
                v = agent.value_network(probes).mean().item()
                self.tb.log_metric(epoch=self.episode, metric_name=f"V_ByDanger/{name}", value=v)

    def _log_policy_entropy(self, agent: "Agent") -> None:
        with torch.no_grad():
            for name, probes in self.probe_states.items():
                logits = agent.policy_network(probes)
                entropy = Categorical(logits=logits).entropy().mean().item()
                self.tb.log_metric(epoch=self.episode, metric_name=f"PolicyEntropy/{name}", value=entropy)

    def _log_visitation(self, agent: "Agent") -> None:
        if self._total_visits == 0:
            return
        for region in ["LowAngle_Safe", "MidAngle_Moderate", "HighAngle_Danger"]:
            count = self._visit_counts.get(region, 0)
            pct = count / self._total_visits
            self.tb.log_metric(epoch=self.episode, metric_name=f"Visitation/{region}", value=pct)
        self._visit_counts = {}
        self._total_visits = 0

    def _log_center_action_distribution(self, agent: "Agent") -> None:
        center = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        with torch.no_grad():
            logits = agent.policy_network(center)
            probs = Categorical(logits=logits).probs
            self.tb.log_metric(epoch=self.episode, metric_name="CenterAction/P_Left", value=probs[0].item())
            self.tb.log_metric(epoch=self.episode, metric_name="CenterAction/P_Right", value=probs[1].item())

    def _log_advantage_by_action(self, agent: "Agent") -> None:
        angles = torch.linspace(-0.3, 0.3, 10, dtype=torch.float64)
        states = torch.stack([torch.tensor([0.0, 0.0, a.item(), 0.0], dtype=torch.float64) for a in angles])
        with torch.no_grad():
            v = agent.value_network(states).squeeze()
            logits = agent.policy_network(states)
            probs = Categorical(logits=logits).probs
            # Weighted mean value by action probability
            left_weight = probs[:, 0].mean().item()
            right_weight = probs[:, 1].mean().item()
            mean_v = v.mean().item()
            self.tb.log_metric(epoch=self.episode, metric_name="ActionBias/MeanP_Left", value=left_weight)
            self.tb.log_metric(epoch=self.episode, metric_name="ActionBias/MeanP_Right", value=right_weight)
            self.tb.log_metric(epoch=self.episode, metric_name="ActionBias/MeanV_AcrossAngles", value=mean_v)

    def _log_value_surface(self, agent: "Agent") -> None:
        if self.episode % self.decision_boundary_freq != 0:
            return

        angles = np.linspace(-0.4, 0.4, 30)
        ang_vels = np.linspace(-0.6, 0.6, 30)
        angle_grid, angvel_grid = np.meshgrid(angles, ang_vels)

        values = np.zeros_like(angle_grid)

        with torch.no_grad():
            for i in range(angle_grid.shape[0]):
                for j in range(angle_grid.shape[1]):
                    state = torch.tensor([0.0, 0.0, angle_grid[i, j], angvel_grid[i, j]], dtype=torch.float64)
                    values[i, j] = agent.value_network(state).item()

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.plot_surface(angle_grid, angvel_grid, values, cmap="viridis")
        ax.set_title(f"V(S) — Episode {self.episode}")
        ax.set_xlabel("Pole Angle")
        ax.set_ylabel("Angular Velocity")
        ax.set_zlabel("V(S)")
        fig.tight_layout()

        self.tb.log_figure(epoch=self.episode, tag="ValueSurface", figure=fig)
        self.tb.flush()

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"value_surface_ep{self.episode}.png")
            fig.savefig(save_path, dpi=100)
        plt.close(fig)
