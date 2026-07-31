from typing import Any

import gymnasium as gym


class TerminalPenaltyWrapper(gym.Wrapper):
    def step(self, action: int) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            reward = -10

        return obs, reward, terminated, truncated, info
