from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from samsara_rl.agent import Agent


class CartPoleLogger:

    def __call__(self, agent: "Agent") -> None:
        if not agent.tensorboard:
            return

        self.episode = agent.curr_episode
        self.tb = agent.tensorboard
        self.get_q = agent.get_q_values

        self._log_probe_leaning()
        self._log_probe_leaning_and_falling()
        self._log_probe_angle_only()
        self._log_q_values()
        self._log_td_error(agent)
        self._log_weights(agent)
        self._log_decision_boundary(agent)

    def _log_probe_leaning(self) -> None:
        q_left = self.get_q(np.array([0.0, 0.0, -0.18, -1.0]))
        self.tb.log_metric(epoch=self.episode, metric_name="ProbeValue/LeaningLeft/PushLeft", value=q_left[0])
        self.tb.log_metric(epoch=self.episode, metric_name="ProbeValue/LeaningLeft/PushRight", value=q_left[1])

    def _log_probe_leaning_and_falling(self) -> None:
        q_left_falling = self.get_q(np.array([0, 0, 0.10, -1.00]))
        q_right_falling = self.get_q(np.array([0, 0, 0.10, 1.00]))

        self.tb.log_metric(epoch=self.episode, metric_name="Probe/LeaningLeftFallingLeft/PushLeft", value=q_left_falling[0])
        self.tb.log_metric(epoch=self.episode, metric_name="Probe/LeaningLeftFallingLeft/PushRight", value=q_left_falling[1])
        self.tb.log_metric(epoch=self.episode, metric_name="Probe/LeaningRightFallingRight/PushLeft", value=q_right_falling[0])
        self.tb.log_metric(epoch=self.episode, metric_name="Probe/LeaningRightFallingRight/PushRight", value=q_right_falling[1])

    def _log_probe_angle_only(self) -> None:
        q_right = self.get_q(np.array([0.0, 0.0, 0.18, 0.0]))
        q_left = self.get_q(np.array([0.0, 0.0, -0.18, 0.0]))

        self.tb.log_metric(epoch=self.episode, metric_name="ProbeAngleOnly/LeaningRight/PushLeft", value=q_right[0])
        self.tb.log_metric(epoch=self.episode, metric_name="ProbeAngleOnly/LeaningRight/PushRight", value=q_right[1])
        self.tb.log_metric(epoch=self.episode, metric_name="ProbeAngleOnly/LeaningLeft/PushLeft", value=q_left[0])
        self.tb.log_metric(epoch=self.episode, metric_name="ProbeAngleOnly/LeaningLeft/PushRight", value=q_left[1])

    def _log_q_values(self) -> None:
        q_centered = self.get_q(np.array([0.0, 0.0, 0.0, 0.0]))
        self.tb.log_metric(epoch=self.episode, metric_name="Q_Centered/PushLeft", value=q_centered[0])
        self.tb.log_metric(epoch=self.episode, metric_name="Q_Centered/PushRight", value=q_centered[1])

        q_left = self.get_q(np.array([0.0, 0.0, -0.18, -1.0]))
        self.tb.log_metric(epoch=self.episode, metric_name="Q_LeaningLeft/PushLeft", value=q_left[0])
        self.tb.log_metric(epoch=self.episode, metric_name="Q_LeaningLeft/PushRight", value=q_left[1])

        q_right = self.get_q(np.array([0.0, 0.0, 0.18, 1.0]))
        self.tb.log_metric(epoch=self.episode, metric_name="Q_LeaningRight/PushLeft", value=q_right[0])
        self.tb.log_metric(epoch=self.episode, metric_name="Q_LeaningRight/PushRight", value=q_right[1])

    def _log_td_error(self, agent: "Agent") -> None:
        if hasattr(agent, "last_td_error"):
            self.tb.log_metric(epoch=self.episode, metric_name="TDError", value=agent.last_td_error)

    def _log_weights(self, agent: "Agent") -> None:
        if not (hasattr(agent, "q") and hasattr(agent.q, "W")):
            return
        w = agent.q.W.data
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                action_name = "Left" if j == 0 else "Right"
                self.tb.log_metric(epoch=self.episode, metric_name=f"Weights/feature{i}_action{action_name}", value=w[i, j])

    def _log_decision_boundary(self, agent: "Agent") -> None:
        if self.episode % 10000 == 0 and hasattr(agent, "log_decision_boundary"):
            agent.log_decision_boundary(self.episode)
