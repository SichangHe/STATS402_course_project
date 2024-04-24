import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from numpy.typing import NDArray
from pettingzoo import ParallelEnv
from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from torch import Tensor

from battlesnake_train.dummy import DummyVecEnv


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
    return np.array(((num,),))


class DynPPO:
    """Dynamic PPO that works with Pettingzoo `ParallelEnv`.
    This object wraps a `stable_baselines3.PPO` instance:

    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    # TODO: Train against older self.

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: ParallelEnv,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        ppo_env = DummyVecEnv(
            1, env.observation_space(env.agents[0]), env.action_space(env.agents[0])
        )
        self.ppo = PPO(
            policy=policy,
            env=ppo_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
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
        `stable_baselines3/common/on_policy_algorithm.py`."""
        iteration = 0

        total_timesteps, callback = self._setup_learn(
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

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> tuple[int, BaseCallback]:
        """Modified from `BaseAlgorithm._setup_learn` in
        `stable_baselines3/common/base_class.py`.

        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.ppo.start_time = time.time_ns()

        if self.ppo.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ppo.ep_info_buffer = deque(maxlen=self.ppo._stats_window_size)
            self.ppo.ep_success_buffer = deque(maxlen=self.ppo._stats_window_size)

        if self.ppo.action_noise is not None:
            self.ppo.action_noise.reset()

        if reset_num_timesteps:
            self.ppo.num_timesteps = 0
            self.ppo._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.ppo.num_timesteps
        self.ppo._total_timesteps = total_timesteps
        self.ppo._num_timesteps_at_start = self.ppo.num_timesteps

        if reset_num_timesteps or len(self._last_observations) == 0:
            self._last_observations = {
                agent: observation[np.newaxis, :].copy()
                for agent, observation in self.env.reset()[0].items()
            }
            for agent in self.agents:
                self._last_episode_starts[agent] = True

            # Retrieve unnormalized observation for saving into the buffer
            if self.ppo._vec_normalize_env is not None:
                self.ppo._last_original_obs = (
                    self.ppo._vec_normalize_env.get_original_obs()
                )

        # Configure logger's outputs if no logger was passed
        if not self.ppo._custom_logger:
            self.ppo._logger = utils.configure_logger(
                self.ppo.verbose,
                self.ppo.tensorboard_log,
                tb_log_name,
                reset_num_timesteps,
            )

        # Create eval callback if needed
        callback = self.ppo._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def collect_rollouts(
        self,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Modified from `OnPolicyAlgorithm.collect_rollouts` in
        `stable_baselines3/common/on_policy_algorithm.py`.

        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
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
                observation = observation[np.newaxis, :].copy()
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
