import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from samsara_rl.control.dyna.deep_dyna_q import DeepDynaQ


# Probe states: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
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


class CartPoleLoggerDyna:
    def __init__(
        self,
        probe_states: dict[str, torch.Tensor] | None = None,
        q_surface_freq: int = 50,
        save_dir: str = "q_surfaces",
    ) -> None:
        self.probe_states = probe_states or {
            "LowAngle_Safe": PROBE_LOW,
            "MidAngle_Moderate": PROBE_MID,
            "HighAngle_Danger": PROBE_HIGH,
        }
        self.q_surface_freq = q_surface_freq
        self.save_dir = save_dir
        self._wm_losses: list[float] = []

    def on_visit(self, agent: "DeepDynaQ") -> None:
        if not agent.tensorboard:
            return
        if hasattr(agent, "_last_wm_loss") and agent._last_wm_loss is not None:
            self._wm_losses.append(agent._last_wm_loss)

    def on_episode(self, agent: "DeepDynaQ") -> None:
        if not agent.tensorboard:
            return

        self.episode = agent.curr_episode
        self.tb = agent.tensorboard

        if self.save_dir and not hasattr(self, "_save_dir_set"):
            self.save_dir = os.path.join(agent.tensorboard.log_dir, self.save_dir)
            self._save_dir_set = True

        self._log_world_model_loss(agent)
        self._log_q_by_danger(agent)
        self._log_q_action_split(agent)
        self._log_epsilon(agent)
        self._log_sim_buffer_size(agent)
        self._log_q_surface(agent)

    def _log_world_model_loss(self, agent: "DeepDynaQ") -> None:
        if not self._wm_losses:
            return
        avg_loss = sum(self._wm_losses) / len(self._wm_losses)
        self.tb.log_metric(epoch=self.episode, metric_name="WorldModel/AvgLoss", value=avg_loss)
        self._wm_losses.clear()

    def _log_q_by_danger(self, agent: "DeepDynaQ") -> None:
        with torch.no_grad():
            for name, probes in self.probe_states.items():
                q_vals = agent.q(probes)
                max_q = q_vals.max(dim=1).values.mean().item()
                self.tb.log_metric(epoch=self.episode, metric_name=f"Q_ByDanger/{name}", value=max_q)

    def _log_q_action_split(self, agent: "DeepDynaQ") -> None:
        center = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        left_lean = torch.tensor([[0.0, 0.0, -0.1, 0.0]], dtype=torch.float64)
        right_lean = torch.tensor([[0.0, 0.0, 0.1, 0.0]], dtype=torch.float64)

        with torch.no_grad():
            for name, state in [("Center", center), ("LeftLean", left_lean), ("RightLean", right_lean)]:
                q_vals = agent.q(state).squeeze()
                self.tb.log_metric(epoch=self.episode, metric_name=f"Q_Action/{name}_Left", value=q_vals[0].item())
                self.tb.log_metric(epoch=self.episode, metric_name=f"Q_Action/{name}_Right", value=q_vals[1].item())

    def _log_epsilon(self, agent: "DeepDynaQ") -> None:
        self.tb.log_metric(epoch=self.episode, metric_name="Exploration/Epsilon", value=agent.search.epsilon)

    def _log_sim_buffer_size(self, agent: "DeepDynaQ") -> None:
        self.tb.log_metric(epoch=self.episode, metric_name="SimBuffer/Size", value=agent.sim_buffer.curr_index)

    def _log_q_surface(self, agent: "DeepDynaQ") -> None:
        if self.episode % self.q_surface_freq != 0:
            return

        angles = np.linspace(-0.4, 0.4, 30)
        ang_vels = np.linspace(-0.6, 0.6, 30)
        angle_grid, angvel_grid = np.meshgrid(angles, ang_vels)

        max_q = np.zeros_like(angle_grid)
        best_action = np.zeros_like(angle_grid)

        with torch.no_grad():
            for i in range(angle_grid.shape[0]):
                for j in range(angle_grid.shape[1]):
                    state = torch.tensor([0.0, 0.0, angle_grid[i, j], angvel_grid[i, j]], dtype=torch.float64)
                    q_vals = agent.q(state)
                    max_q[i, j] = q_vals.max().item()
                    best_action[i, j] = q_vals.argmax().item()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={"projection": "3d"})

        axes[0].plot_surface(angle_grid, angvel_grid, max_q, cmap="viridis")
        axes[0].set_title(f"max Q(S,a) — Episode {self.episode}")
        axes[0].set_xlabel("Pole Angle")
        axes[0].set_ylabel("Angular Velocity")
        axes[0].set_zlabel("Q")

        axes[1].plot_surface(angle_grid, angvel_grid, best_action, cmap="coolwarm")
        axes[1].set_title(f"Best Action — Episode {self.episode}")
        axes[1].set_xlabel("Pole Angle")
        axes[1].set_ylabel("Angular Velocity")
        axes[1].set_zlabel("Action (0=L, 1=R)")

        fig.tight_layout()
        self.tb.log_figure(epoch=self.episode, tag="QSurface", figure=fig)
        self.tb.flush()

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"q_surface_ep{self.episode}.png")
            fig.savefig(save_path, dpi=100)
        plt.close(fig)
