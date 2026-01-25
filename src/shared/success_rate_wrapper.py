from typing import SupportsFloat, Any
import gymnasium as gym
from collections import deque
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType


class SuccessRateWrapper(gym.Wrapper):
    """Gym wrapper that tracks and computes the success rate of episodes"""

    def __init__(self, env: gym.Env, window_size: int = 50):
        super().__init__(env)
        self.success_history = deque(maxlen=window_size)
        self.current_success_rate = 0

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            is_success = info.get("is_success")
            self.success_history.append(float(is_success))
            self.current_success_rate = np.mean(self.success_history)
            info["success_rate"] = self.current_success_rate

        return observation, reward, terminated, truncated, info
