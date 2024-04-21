from pettingzoo.test import parallel_api_test

from battlesnake_gym import BattlesnakeEnv


def test_parallel_api() -> None:
    env = BattlesnakeEnv()
    parallel_api_test(env, num_cycles=1_000)

