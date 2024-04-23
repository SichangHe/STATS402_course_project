import numpy as np
import torch as th
from gymnasium import spaces
from pettingzoo import ParallelEnv
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor


class DynPPO:
    """Dynamic PPO that works with Pettingzoo `ParallelEnv`."""

    # TODO: Train against older self.

    env: ParallelEnv

    def __init__(self, ppo: PPO):
        self.ppo = ppo
        assert isinstance(ppo.env, ParallelEnv)
        self.env = ppo.env

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
                self.env,
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
        # TODO: Adapt to `ParallelEnv`.
        assert self.ppo._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.ppo.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.ppo.use_sde:
            self.ppo.policy.reset_noise(self.ppo.batch_size)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.ppo.use_sde
                and self.ppo.sde_sample_freq > 0
                and n_steps % self.ppo.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.ppo.policy.reset_noise(self.ppo.batch_size)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self.ppo._last_obs, self.ppo.device)
                actions, values, log_probs = self.ppo.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.ppo.action_space, spaces.Box):
                if self.ppo.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.ppo.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.ppo.action_space.low, self.ppo.action_space.high
                    )

            # FIXME: Match `ParallelEnv` API.
            new_obs, rewards, dones, infos = self.env.step(clipped_actions)

            self.ppo.num_timesteps += self.env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self.ppo._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.ppo.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.ppo.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.ppo.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.ppo.gamma * terminal_value

            rollout_buffer.add(
                self.ppo._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self.ppo._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self.ppo._last_obs = new_obs  # type: ignore[assignment]
            self.ppo._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.ppo.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
