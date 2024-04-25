from typing import Generator, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from numpy.typing import NDArray
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples


class GrowableRolloutBuffer(RolloutBuffer):
    """A rollout buffer that can grow over the specified buffer size."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.target_buffer_size = buffer_size
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            gae_lambda=gae_lambda,
            gamma=gamma,
        )

    def reset(self) -> None:
        # Pre-allocate more, anticipating to run over.
        self.true_buffer_size = self.target_buffer_size * 2
        self.buffer_size = self.true_buffer_size
        super().reset()
        self.buffer_size = self.target_buffer_size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        If the buffer is already full, grows the buffer by 2x.

        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if self.full:
            self._grow(self.buffer_size + 1)
            self.buffer_size += 1
        super().add(obs, action, reward, episode_start, value, log_prob)

    def _grow(self, new_min_size: int):
        while self.true_buffer_size < new_min_size:
            self.observations = double_array(self.observations)
            self.actions = double_array(self.actions)
            self.rewards = double_array(self.rewards)
            self.returns = double_array(self.returns)
            self.episode_starts = double_array(self.episode_starts)
            self.values = double_array(self.values)
            self.log_probs = double_array(self.log_probs)
            self.advantages = double_array(self.advantages)
            self.true_buffer_size *= 2

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        self._clip_empty()
        return super().get(batch_size)

    def _clip_empty(self):
        self.observations = self.observations[: self.buffer_size]
        self.actions = self.actions[: self.buffer_size]
        self.rewards = self.rewards[: self.buffer_size]
        self.returns = self.returns[: self.buffer_size]
        self.episode_starts = self.episode_starts[: self.buffer_size]
        self.values = self.values[: self.buffer_size]
        self.log_probs = self.log_probs[: self.buffer_size]
        self.advantages = self.advantages[: self.buffer_size]
        self.true_buffer_size = self.buffer_size


def double_array(array: NDArray):
    return np.concatenate([array, np.zeros_like(array)])
