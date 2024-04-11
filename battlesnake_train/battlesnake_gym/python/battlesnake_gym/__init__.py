from typing import Any

import numpy as np
from gymnasium import Env

from battlesnake_gym._lowlevel import SnakeGame, hello


class BattlesnakeEnv(Env):
    def __init__(self):
        self.snake_game = SnakeGame()

    # override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        # TODO: Use seed and options.
        self.snake_game.reset()
        # TODO: Implement.
        observation = np.zeros((1, 1))
        info: dict[str, Any] = {}
        return observation, info

    # override
    def step(self, action):
        # TODO: Implement.
        observation = np.zeros((1, 1))
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        return observation, reward, terminated, truncated, info

    # override
    def render(self):
        # TODO: Implement.
        pass

    # override
    def close(self):
        pass


__all__ = ["hello"]
