from typing import Any, Final

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from supersuit.multiagent_wrappers.black_death import black_death_par
from supersuit.vector import MarkovVectorEnv
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
        rewards = {i: raw_rewards[i] for i in self.agents}
        terminations = {i: raw_terminations[i] for i in self.agents}
        truncations = {i: False for i in self.agents}
        observations = self._observations()
        self.agents = [i for i in self.agents if not raw_terminations[i]]
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
            if i in self.agents
        }

    def _make_infos(self) -> dict[int, dict[str, Any]]:
        return {i: {} for i in self.agents}


def convert_action(action: int | None) -> int:
    if action is None:
        return 0
    return action - 1


def gymnasium_vector_env() -> MarkovVectorEnv:
    battlesnake_env = BattlesnakeEnv()
    black_death_env = black_death_par(battlesnake_env)
    markov_vector_env = MarkovVectorEnv(black_death_env)
    return markov_vector_env


class VecEnvWrapper(SB3VecEnvWrapper):
    """Class to override nonsense in `SB3VecEnvWrapper`."""

    venv: MarkovVectorEnv

    # Override
    def reset(self, seed=None, options=None):
        observations, self.reset_infos = self.venv.reset(seed=seed, options=options)
        return observations

    # Override
    def render(self, mode=None):
        if mode:
            self.venv.par_env.render_mode = mode
        return self.venv.par_env.render()


def sb3_vec_env() -> VecEnvWrapper:
    markov_vector_env = gymnasium_vector_env()
    sb3_vec_env = VecEnvWrapper(markov_vector_env)
    return sb3_vec_env


__all__ = ["hello", "gymnasium_vector_env", "VecEnvWrapper", "sb3_vec_env"]
