from dataclasses import dataclass

import numpy as np
import torch as th
from gymnasium import spaces
from numpy.typing import NDArray
from pettingzoo import ParallelEnv
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor
from torch import Tensor


@dataclass
class RolloutBufferCacheItem:
    observation: NDArray
    action: NDArray
    reward: NDArray | None
    episode_start: bool
    value: Tensor
    log_prob: Tensor

    def add_to_buffer(self, rollout_buffer: RolloutBuffer):
        assert self.reward is not None
        rollout_buffer.add(
            self.observation,
            self.action,
            self.reward,
            matrix1x1(self.episode_start),
            self.value,
            self.log_prob,
        )


def matrix1x1(num):
    return np.array((num,))[np.newaxis, :]


class DynPPO:
    """Dynamic PPO that works with Pettingzoo `ParallelEnv`.
    The dummy `ppo` fed in should have its `env` be `None`."""

    # TODO: Train against older self.

    def __init__(self, ppo: PPO, env: ParallelEnv):
        self.ppo = ppo
        self.env = env
        # Note: Should be `AgentID` not `int`, but who cares.
        self.agents: list[int] = env.possible_agents
        self.rollout_buffer_cache: dict[int, list[RolloutBufferCacheItem]] = {
            a: [] for a in self.agents
        }
        self._last_observations: dict[int, NDArray] = {}
        self._last_episode_starts: dict[int, bool] = {a: True for a in self.agents}

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """Modified from `OnPolicyAlgorithm.learn` in
        `stable_baselines3/common/on_policy_algorithm.py`"""
        iteration = 0

        # FIXME: Replace these with custom code.
        total_timesteps, callback = self.ppo._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.ppo.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                callback,
                self.ppo.rollout_buffer,
                n_rollout_steps=self.ppo.n_steps,
            )

            if not continue_training:
                break

            iteration += 1
            self.ppo._update_current_progress_remaining(
                self.ppo.num_timesteps, total_timesteps
            )

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ppo.ep_info_buffer is not None
                self.ppo._dump_logs(iteration)

            self.ppo.train()

        callback.on_training_end()

        return self

    def collect_rollouts(
        self,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Modified from `OnPolicyAlgorithm.collect_rollouts` in
        `stable_baselines3/common/on_policy_algorithm.py`"""
        assert len(self._last_observations) > 0, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.ppo.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.ppo.use_sde:
            self.ppo.policy.reset_noise(self.ppo.batch_size)

        callback.on_rollout_start()

        last_observation, dones = None, None
        while n_steps < n_rollout_steps:
            if (
                self.ppo.use_sde
                and self.ppo.sde_sample_freq > 0
                and n_steps % self.ppo.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.ppo.policy.reset_noise(self.ppo.batch_size)

            clipped_action_map: dict[int, NDArray] = {}
            for agent, observation in self._last_observations.items():
                with th.no_grad():
                    obs_tensor = obs_as_tensor(observation, self.ppo.device)
                    action_tensor, value, log_prob = self.ppo.policy(obs_tensor)
                action: NDArray = action_tensor.cpu().numpy()

                # Rescale and perform action
                clipped_action = action

                if isinstance(self.ppo.action_space, spaces.Box):
                    if self.ppo.policy.squash_output:
                        # Unscale the actions to match env bounds
                        # if they were previously squashed (scaled in [-1, 1])
                        clipped_action = self.ppo.policy.unscale_action(clipped_action)
                    else:
                        # Otherwise, clip the actions to avoid out of bound error
                        # as we are sampling from an unbounded Gaussian distribution
                        clipped_action = np.clip(
                            clipped_action,
                            self.ppo.action_space.low,
                            self.ppo.action_space.high,
                        )

                clipped_action_map[agent] = clipped_action[0]

                if isinstance(self.ppo.action_space, spaces.Discrete):
                    # Reshape in case of discrete action
                    action = action.reshape(-1, 1)
                item = RolloutBufferCacheItem(
                    observation,
                    action,
                    None,
                    self._last_episode_starts[agent],
                    value,
                    log_prob,
                )
                self.rollout_buffer_cache[agent].append(item)

            new_obs, raw_rewards, terminations, truncations, infos = self.env.step(
                clipped_action_map
            )
            rewards: dict[int, NDArray] = {
                a: matrix1x1(reward) for a, reward in raw_rewards.items()
            }
            dones = np.asarray(
                [
                    terminations.get(agent, False) or truncations.get(agent, False)
                    for agent in self.agents
                ]
            )
            info_list = [infos.get(agent, {}) for agent in self.agents]

            self.ppo.num_timesteps += len(clipped_action_map)

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self.ppo._update_info_buffer(info_list, dones)
            n_steps += 1

            self._last_observations.clear()
            for agent, observation in new_obs.items():
                observation = observation[np.newaxis, :]
                self._last_observations[agent] = observation
                done = terminations[agent] or truncations[agent]
                self._last_episode_starts[agent] = done
                # Handle timeout by bootstraping with value function
                if (
                    done
                    and infos[agent].get("terminal_observation") is not None
                    and infos[agent].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.ppo.policy.obs_to_tensor(
                        infos[agent]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.ppo.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[agent] += self.ppo.gamma * terminal_value

                self.rollout_buffer_cache[agent][-1].reward = rewards[agent]
                if done:
                    # Dump out rollout buffer cache if this agent is done.
                    for item in self.rollout_buffer_cache[agent]:
                        item.add_to_buffer(rollout_buffer)
                    self.rollout_buffer_cache[agent].clear()
                    last_observation = observation

        with th.no_grad():
            # Compute value for the last timestep
            assert last_observation is not None
            values = self.ppo.policy.predict_values(
                obs_as_tensor(last_observation, self.ppo.device)
            )

        assert dones is not None
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
