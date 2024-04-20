from typing import Any, Final

import numpy as np
import supersuit as ss
from gymnasium import spaces
from pettingzoo import ParallelEnv

from battlesnake_gym._lowlevel import SnakeGame, hello

BOARD_SIZE: Final = 11
N_SNAKES: Final = 4

PADDED_SIZE: Final = BOARD_SIZE * 2 - 1
N_LAYERS: Final = 10

OBSERVATION_SPACE: Final = spaces.Box(-1.0, 1.0, (N_LAYERS, PADDED_SIZE, PADDED_SIZE))
ACTION_SPACE: Final = spaces.Discrete(4)


class BattlesnakeEnv(ParallelEnv):
    metadata = {
        # TODO: render_modes
        "render_modes": [],
    }

    # TODO: Allow training with older self.
    def __init__(self):
        self.snake_game = SnakeGame()

    # override
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        # TODO: Use seed and options.
        _ = options
        self.snake_game.reset()
        # TODO: Implement.
        observations = self._observations()
        infos = self._make_infos()
        return observations, infos

    # override
    def step(self, actions: dict[int, int]):
        actions_converted = [actions[i] for i in range(N_SNAKES)]
        raw_rewards, raw_terminations = self.snake_game.step(actions_converted)
        infos = self._make_infos()
        rewards = {i: raw_rewards[i] for i in range(N_SNAKES)}
        terminations = {i: raw_terminations[i] for i in range(N_SNAKES)}
        truncations = {i: False for i in range(N_SNAKES)}
        observations = self._observations()
        return observations, rewards, terminations, truncations, infos

    # override
    def render(self):
        # TODO: Implement.
        pass

    # override
    def observation_space(self, agent: int):
        _ = agent
        return OBSERVATION_SPACE

    # override
    def action_space(self, agent: int):
        _ = agent
        return ACTION_SPACE

    def _observations(self):
        # TODO: Get from self.game.
        return {i: np.zeros((1, 1), dtype=np.float32) for i in range(N_SNAKES)}

    def _make_infos(self) -> dict[int, dict[str, Any]]:
        return {i: {} for i in range(N_SNAKES)}


def make_battlesnake_env():
    env = BattlesnakeEnv()
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, base_class="stable_baselines3")
    return env


__all__ = ["hello", "make_battlesnake_env"]