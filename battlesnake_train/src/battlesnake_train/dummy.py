from gymnasium import Space
from stable_baselines3.common.vec_env import VecEnv


class DummyVecEnv(VecEnv):
    """Does not actually implement the interface, just to provide some fields."""

    def __init__(self, num_envs: int, observation_space: Space, action_space: Space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):  # type: ignore[reportIncompatibleMethodOverride]
        pass

    def step_async(self):  # type: ignore[reportIncompatibleMethodOverride]
        pass

    def step_wait(self):  # type: ignore[reportIncompatibleMethodOverride]
        pass

    def close(self):
        pass

    def get_attr(self):  # type: ignore[reportIncompatibleMethodOverride]
        pass

    def set_attr(self):  # type: ignore[reportIncompatibleMethodOverride]
        pass

    def env_method(self):  # type: ignore[reportIncompatibleMethodOverride]
        pass

    def env_is_wrapped(self):  # type: ignore[reportIncompatibleMethodOverride]
        pass
