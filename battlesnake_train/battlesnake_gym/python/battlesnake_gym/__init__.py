from typing import Any, Final

import numpy as np
import supersuit as ss
from gymnasium import spaces
from pettingzoo import ParallelEnv
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from battlesnake_gym._lowlevel import SnakeGame, hello

BOARD_SIZE: Final = 11
N_SNAKES: Final = 4

# Clockwise, opposite to `np.rot90`.
UP: Final = 0
RIGHT: Final = 1
DOWN: Final = 2
LEFT: Final = 3

PADDED_SIZE: Final = BOARD_SIZE * 2 - 1
N_LAYERS: Final = 9

OBSERVATION_SPACE: Final = spaces.Box(-1.0, 1.0, (N_LAYERS, PADDED_SIZE, PADDED_SIZE))
ACTION_SPACE: Final = spaces.Discrete(3)


class BattlesnakeEnv(ParallelEnv):
    metadata = {
        "name": "Battlesnake",
        "render_modes": ["ansi"],
    }

    # TODO: Allow training with older self.
    def __init__(self):
        self.snake_game = SnakeGame()
        self.possible_agents = list(range(N_SNAKES))
        self.agents = list(range(N_SNAKES))
        self.render_mode = "ansi"

    # override
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.agents = list(range(N_SNAKES))
        # TODO: Use seed and options.
        _ = seed, options
        self.snake_game.reset()
        observations = self._observations()
        infos = self._make_infos()
        return observations, infos

    # override
    def step(self, actions: dict[int, int]):
        actions_converted = [convert_action(actions.get(i)) for i in range(N_SNAKES)]
        raw_rewards, raw_terminations = self.snake_game.step(actions_converted)
        infos = self._make_infos()
        rewards = {i: raw_rewards[i] for i in range(N_SNAKES)}
        self.agents = [i for i in self.agents if not raw_terminations[i]]
        terminations = {i: raw_terminations[i] for i in range(N_SNAKES)}
        truncations = {i: False for i in range(N_SNAKES)}
        observations = self._observations()
        return observations, rewards, terminations, truncations, infos

    # override
    def render(self) -> str:
        return self.snake_game.render()

    # override
    def observation_space(self, agent: int):
        _ = agent
        return OBSERVATION_SPACE

    # override
    def action_space(self, agent: int):
        _ = agent
        return ACTION_SPACE

    def _observations(self):
        states, snake_facings = self.snake_game.states()
        return {
            i: np.rot90(state, k=facing, axes=(1, 2))
            for i, (state, facing) in enumerate(zip(states, snake_facings))
        }

    def _make_infos(self) -> dict[int, dict[str, Any]]:
        return {i: {} for i in range(N_SNAKES)}


def convert_action(action: int | None) -> int:
    if action is None:
        return 0
    return action - 1


def make_battlesnake_env() -> SB3VecEnvWrapper:
    env = BattlesnakeEnv()
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # Work around nonsense in SuperSuit.
    env.seed = placeholder_seed  # type: ignore
    env.render = replacement_render
    env = ss.concat_vec_envs_v1(env, 4, base_class="stable_baselines3")
    return env


def placeholder_seed(env, seed=None):
    _ = env, seed


def replacement_render(env, mode=None):
    _ = mode
    return env.venv.render()


__all__ = ["hello", "make_battlesnake_env"]
