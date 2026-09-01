import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from samsara_rl.agent import Agent


class CartPoleLogger:
    def __init__(self, decision_boundary_freq: int = 20, save_dir: str = "decision_surfaces") -> None:
        self.decision_boundary_freq = decision_boundary_freq
        self.save_dir = save_dir

    def on_visit(self, agent: "Agent") -> None:
        if not agent.tensorboard:
            return

        self.tb = agent.tensorboard
        self.step_count = getattr(self, "step_count", 0) + 1

        self._log_td_target(agent)
        self._log_td_error(agent)
        self._log_probe_convergence(agent)
        self._log_loss(agent)

    def on_episode(self, agent: "Agent") -> None:
        if not agent.tensorboard:
            return

        self.episode = agent.curr_episode
        self.tb = agent.tensorboard
        self.get_q = agent.get_q_values

        if self.save_dir and not hasattr(self, "_save_dir_set"):
            self.save_dir = os.path.join(agent.tensorboard.log_dir, self.save_dir)
            self._save_dir_set = True

        self._log_probe_leaning()
        self._log_probe_leaning_and_falling()
        self._log_probe_angle_only()
        self._log_q_values()
        self._log_weights(agent)
        self._log_decision_boundary(agent)

    def _log_probe_leaning(self) -> None:
        q_left = self.get_q(np.array([0.0, 0.0, -0.72, -0.33]))
        self.tb.log_metric(
            epoch=self.episode,
            metric_name="ProbeValue/LeaningLeft/PushLeft",
            value=q_left[0],
        )
        self.tb.log_metric(
            epoch=self.episode,
            metric_name="ProbeValue/LeaningLeft/PushRight",
            value=q_left[1],
        )

    def _log_probe_leaning_and_falling(self) -> None:
        q_left_falling = self.get_q(np.array([0, 0, -0.48, -0.33]))
        q_right_falling = self.get_q(np.array([0, 0, 0.48, 0.33]))

        self.tb.log_metric(
            epoch=self.episode,
            metric_name="Probe/LeaningLeftFallingLeft/PushLeft",
            value=q_left_falling[0],
        )
        self.tb.log_metric(
            epoch=self.episode,
            metric_name="Probe/LeaningLeftFallingLeft/PushRight",
            value=q_left_falling[1],
        )
        self.tb.log_metric(
            epoch=self.episode,
            metric_name="Probe/LeaningRightFallingRight/PushLeft",
            value=q_right_falling[0],
        )
        self.tb.log_metric(
            epoch=self.episode,
            metric_name="Probe/LeaningRightFallingRight/PushRight",
            value=q_right_falling[1],
        )

    def _log_td_target(self, agent: "Agent") -> None:
        if not hasattr(agent, "td_target_range") or not agent.td_target_range:
            return
        self.tb.log_metric(epoch=self.step_count, metric_name="TD_Target/Min", value=agent.td_target_range[0])
        self.tb.log_metric(epoch=self.step_count, metric_name="TD_Target/Max", value=agent.td_target_range[1])

    def _log_probe_angle_only(self) -> None:
        q_right = self.get_q(np.array([0.0, 0.0, 0.72, 0.0]))
        q_left = self.get_q(np.array([0.0, 0.0, -0.72, 0.0]))

        self.tb.log_metric(
            epoch=self.episode,
            metric_name="ProbeAngleOnly/LeaningRight/PushLeft",
            value=q_right[0],
        )
        self.tb.log_metric(
            epoch=self.episode,
            metric_name="ProbeAngleOnly/LeaningRight/PushRight",
            value=q_right[1],
        )
        self.tb.log_metric(
            epoch=self.episode,
            metric_name="ProbeAngleOnly/LeaningLeft/PushLeft",
            value=q_left[0],
        )
        self.tb.log_metric(
            epoch=self.episode,
            metric_name="ProbeAngleOnly/LeaningLeft/PushRight",
            value=q_left[1],
        )

    def _log_q_values(self) -> None:
        q_centered = self.get_q(np.array([0.0, 0.0, 0.0, 0.0]))
        self.tb.log_metric(epoch=self.episode, metric_name="Q_Centered/PushLeft", value=q_centered[0])
        self.tb.log_metric(epoch=self.episode, metric_name="Q_Centered/PushRight", value=q_centered[1])

        q_left = self.get_q(np.array([0.0, 0.0, -0.72, -0.33]))
        self.tb.log_metric(epoch=self.episode, metric_name="Q_LeaningLeft/PushLeft", value=q_left[0])
        self.tb.log_metric(epoch=self.episode, metric_name="Q_LeaningLeft/PushRight", value=q_left[1])

        q_right = self.get_q(np.array([0.0, 0.0, 0.72, 0.33]))
        self.tb.log_metric(epoch=self.episode, metric_name="Q_LeaningRight/PushLeft", value=q_right[0])
        self.tb.log_metric(epoch=self.episode, metric_name="Q_LeaningRight/PushRight", value=q_right[1])

    def _log_td_error(self, agent: "Agent") -> None:
        if hasattr(agent, "last_td_error"):
            self.tb.log_metric(epoch=self.step_count, metric_name="TDError", value=agent.last_td_error)

    def _log_loss(self, agent: "Agent") -> None:
        if hasattr(agent, "last_loss"):
            self.tb.log_metric(epoch=self.step_count, metric_name="Loss", value=agent.last_loss)

    def _log_probe_convergence(self, agent: "Agent") -> None:
        if not hasattr(agent, "q") or not hasattr(agent, "target_network"):
            return
        probe = np.array([0, 0, 0.48, 0.33])
        with torch.no_grad():
            prediction = agent.q(probe)
            target = agent.target_network(probe)
        self.tb.log_metrics(
            epoch=self.step_count,
            metric_name="ProbeConvergence/PushLeft",
            values={"Q": prediction[0], "Target": target[0]},
        )
        self.tb.log_metrics(
            epoch=self.step_count,
            metric_name="ProbeConvergence/PushRight",
            values={"Q": prediction[1], "Target": target[1]},
        )

    def _log_weights(self, agent: "Agent") -> None:
        if not (hasattr(agent, "q") and hasattr(agent.q, "W")):
            return
        w = agent.q.W.data
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                action_name = "Left" if j == 0 else "Right"
                self.tb.log_metric(
                    epoch=self.episode,
                    metric_name=f"Weights/feature{i}_action{action_name}",
                    value=w[i, j],
                )

    def _log_decision_boundary(self, agent: "Agent") -> None:
        if self.episode % self.decision_boundary_freq != 0:
            return

        angles = np.linspace(-1.2, 1.2, 30)
        ang_vels = np.linspace(-1.2, 1.2, 30)
        angle_grid, angvel_grid = np.meshgrid(angles, ang_vels)

        q_left = np.zeros_like(angle_grid)
        q_right = np.zeros_like(angle_grid)

        for i in range(angle_grid.shape[0]):
            for j in range(angle_grid.shape[1]):
                state = np.array([0.0, 0.0, angle_grid[i, j], angvel_grid[i, j]])
                q_vals = self.get_q(state)
                q_left[i, j] = q_vals[0]
                q_right[i, j] = q_vals[1]

        fig = plt.figure(figsize=(14, 5))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(angle_grid, angvel_grid, q_left, cmap="viridis")
        ax1.set_title("Q(PushLeft)")
        ax1.set_xlabel("Angle")
        ax1.set_ylabel("Angular Velocity")
        ax1.set_zlabel("Q Value")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(angle_grid, angvel_grid, q_right, cmap="viridis")
        ax2.set_title("Q(PushRight)")
        ax2.set_xlabel("Angle")
        ax2.set_ylabel("Angular Velocity")
        ax2.set_zlabel("Q Value")

        fig.suptitle(f"Decision Surface — Episode {self.episode}")
        fig.tight_layout()

        self.tb.log_figure(epoch=self.episode, tag="DecisionSurface", figure=fig)
        self.tb.flush()

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(self.save_dir, f"decision_surface_ep{self.episode}.png"),
                dpi=100,
            )

        plt.close(fig)

        # Action advantage: Q(PushLeft) - Q(PushRight)
        advantage = q_left - q_right

        fig2 = plt.figure(figsize=(8, 6))
        ax = fig2.add_subplot(1, 1, 1, projection="3d")
        ax.plot_surface(angle_grid, angvel_grid, advantage, cmap="RdBu")
        ax.set_title("Q(PushLeft) - Q(PushRight)")
        ax.set_xlabel("Angle")
        ax.set_ylabel("Angular Velocity")
        ax.set_zlabel("Advantage")

        fig2.suptitle(f"Action Advantage — Episode {self.episode}")
        fig2.tight_layout()

        self.tb.log_figure(epoch=self.episode, tag="ActionAdvantage", figure=fig2)
        self.tb.flush()

        if self.save_dir:
            fig2.savefig(
                os.path.join(self.save_dir, f"action_advantage_ep{self.episode}.png"),
                dpi=100,
            )

        plt.close(fig2)
