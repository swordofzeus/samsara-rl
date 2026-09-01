from typing import Any

import gymnasium as gym
import numpy as np

POSITION_RANGE = [-2.4, 2.4]
VELOCITY_RANGE = [-3.0, 3.0]
ANGLE_RANGE = [-0.209, 0.209]
ANGLE_VEL_RANGE = [-3.0, 3.0]


class ScaledCartPole(gym.Wrapper):
    def scale_value(self, value: float, value_range: list[float]) -> float:
        value = np.clip(value, value_range[0], value_range[1])
        return 2 * (value - value_range[0]) / (value_range[1] - value_range[0]) - 1

    def scale_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.array([
            self.scale_value(obs[0], POSITION_RANGE),
            self.scale_value(obs[1], VELOCITY_RANGE),
            self.scale_value(obs[2], ANGLE_RANGE),
            self.scale_value(obs[3], ANGLE_VEL_RANGE),
        ])

    def step(self, action: int) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.scale_obs(obs)
        if terminated:
            reward = -10

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self.scale_obs(obs), info
