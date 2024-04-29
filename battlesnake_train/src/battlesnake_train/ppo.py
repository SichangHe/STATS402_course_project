import io
import pathlib
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from numpy.typing import NDArray
from pettingzoo import ParallelEnv
from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    ConvertCallback,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from torch import Tensor

from battlesnake_train.buf import GrowableRolloutBuffer
from battlesnake_train.disk import all_prev_models, find_last_model
from battlesnake_train.dummy import DummyVecEnv
from battlesnake_train.progress import DynPPOProgressBarCallback


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

    Additional parameters:

    :param finish_episode: Whether to finish current episodes before training.
    :param p_use_older_version: Probability to use an older version of the
    model.

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

    def __init__(
        self,
        policy: str | Type[ActorCriticPolicy] | None,
        env: ParallelEnv,
        finish_episode: bool = True,
        p_use_older_version: float = 0.2,
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
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = GrowableRolloutBuffer,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        trial_index: int = -1,
        save_model_name: str = "dyn-ppo",
        saved_model_regex: re.Pattern | None = None,
        ppo: PPO | None = None,
    ):
        if policy is None:
            assert (
                ppo is not None
            ), "At least one of `policy` and `ppo` must be provided to initialize `DynPPO`!"
            self.ppo = ppo
        else:
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
        self.finish_episode = finish_episode
        self.p_use_older_version = p_use_older_version
        self.save_model_name = save_model_name
        self.saved_model_regex = saved_model_regex or re.compile(
            save_model_name + r"(\d+)\.model"
        )
        self.trial_index = trial_index
        # Note: Should be `AgentID` not `int`, but who cares.
        self.agents: list[int] = env.possible_agents
        self.rollout_buffer_cache: dict[int, list[RolloutBufferCacheItem]] = {
            a: [] for a in self.agents
        }
        self._last_observations: dict[int, NDArray] = {}
        self._last_episode_starts: dict[int, bool] = {a: True for a in self.agents}
        self.prev_trials: list[int] = []
        self.prev_models: dict[int, DynPPO | str | pathlib.Path | io.BufferedIOBase] = (
            {}
        )
        self.prev_trials_picked: list[list[int]] = []

    def learn_trials(
        self,
        n_trial: int,
        n_timestep_per_trial: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ):
        """Like `DynPPO.learn`, but for multiple trials, saving intermediate
        models along the way.
        """
        self._find_prev_models()

        for trial in range(self.trial_index + 1, self.trial_index + 1 + n_trial):
            self.trial_index = trial
            self.learn(
                n_timestep_per_trial,
                callback,
                log_interval,
                tb_log_name,
                reset_num_timesteps,
                progress_bar,
            )
            time_took = time.time_ns() - self.ppo.start_time
            print(f"Trial {trial} took {time_took / 1e9} seconds.")
            saved_self_path = self.save(exclude=exclude, include=include)
            self.prev_trials.append(trial)
            self.prev_models[trial] = saved_self_path

    def _find_prev_models(self):
        for trial, model_file in all_prev_models(self.saved_model_regex):
            if trial not in self.prev_models:
                self.prev_trials.append(trial)
                self.prev_models[trial] = model_file

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
            self._reset_episode()

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

    def _reset_episode(self):
        self._last_observations = {
            agent: observation[np.newaxis, :].copy()
            for agent, observation in self.env.reset()[0].items()
        }
        for agent in self.agents:
            self._last_episode_starts[agent] = True
            self.rollout_buffer_cache[agent].clear()

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        """Modified
        `stable_baselines3.common.base_class.BaseAlgorithm._init_callback`
        to use the corrected progress bar implementation.

        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, DynPPOProgressBarCallback()])

        callback.init_callback(self.ppo)
        return callback

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

        dyn_ppo_agents = self._pick_dyn_ppo_agents()

        episode_ongoing = False
        """To finish the current episode before training."""
        last_observation, done_matrix = None, None
        while episode_ongoing or n_steps < n_rollout_steps:
            episode_ongoing = self.finish_episode
            if (
                self.ppo.use_sde
                and self.ppo.sde_sample_freq > 0
                and n_steps % self.ppo.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.ppo.policy.reset_noise(self.ppo.batch_size)

            clipped_action_map: dict[int, NDArray] = {}
            for agent, observation in self._last_observations.items():
                dyn_ppo = dyn_ppo_agents.get(agent, self)
                with th.no_grad():
                    obs_tensor = obs_as_tensor(observation, self.ppo.device)
                    action_tensor, value, log_prob = dyn_ppo.ppo.policy(obs_tensor)
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

                if agent not in dyn_ppo_agents:
                    # This agent is `self`.
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

            new_obs, rewards, terminations, truncations, infos = self.env.step(
                clipped_action_map
            )
            self.ppo.num_timesteps += len(clipped_action_map)

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._last_observations.clear()
            for agent, observation in new_obs.items():
                observation = observation[np.newaxis, :].copy()
                done = terminations[agent] or truncations[agent]
                self._last_episode_starts[agent] = done
                done_matrix = matrix1x1(done)
                if not done:
                    # No need to record observations for done agents.
                    self._last_observations[agent] = observation

                self.ppo._update_info_buffer([infos[agent]], done_matrix)

                if agent not in dyn_ppo_agents:
                    # This agent is `self`.
                    reward = matrix1x1(rewards[agent])
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
                        reward += self.ppo.gamma * terminal_value

                    cache = self.rollout_buffer_cache[agent]
                    cache[-1].reward = reward
                    if done:
                        # Dump out rollout buffer cache if this agent is done.
                        # Assuming the buffer grows if `finish_episode`.
                        buffer_free_size = (
                            len(cache)
                            if self.finish_episode
                            else (rollout_buffer.buffer_size - rollout_buffer.size())
                        )
                        for item in cache[:buffer_free_size]:
                            item.add_to_buffer(rollout_buffer)
                        if buffer_free_size >= len(cache):
                            n_steps += len(cache)
                            cache.clear()
                            last_observation = observation
                        else:
                            n_steps += buffer_free_size
                            cache = cache[buffer_free_size:]
                            last_observation = cache[0].observation

            if all(
                (
                    # Either every agent is done, or is not `self`.
                    last_episode_start or (agent in dyn_ppo_agents)
                    for agent, last_episode_start in self._last_episode_starts.items()
                )
            ):
                # All agents are either done, or is not `self`. Reset episode.
                self._reset_episode()
                dyn_ppo_agents = self._pick_dyn_ppo_agents()
                episode_ongoing = False

        with th.no_grad():
            # Compute value for the last timestep
            assert last_observation is not None
            values = self.ppo.policy.predict_values(
                obs_as_tensor(last_observation, self.ppo.device)
            )

        assert done_matrix is not None
        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=done_matrix
        )

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def _pick_dyn_ppo_agents(self):
        """Return a dictionary from agent IDs to `DynPPO` instances.
        Absence of an agent ID means to use `self`."""
        trials_picked: list[int] = []
        agents: dict[int, DynPPO] = {}
        if len(self.prev_models) > 0:
            while True:
                for agent in self.agents:
                    if random.random() < self.p_use_older_version:
                        trial = random.choice(self.prev_trials)
                        trials_picked.append(trial)
                        maybe_dyn_ppo = self.prev_models[trial]
                        if isinstance(maybe_dyn_ppo, DynPPO):
                            agents[agent] = maybe_dyn_ppo
                        else:
                            dyn_ppo = DynPPO.load(
                                maybe_dyn_ppo,
                                self.env,
                                self.ppo.device,
                                force_reset=False,
                            )
                            self.prev_models[agent] = dyn_ppo
                            agents[agent] = dyn_ppo

                if len(agents) < len(self.agents):
                    break
                # Else: no agent is `self`, re-pick.
                trials_picked.clear()
                agents.clear()
        self.prev_trials_picked.append(trials_picked)
        return agents

    def predict(
        self,
        observations: dict[int, NDArray],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[dict[int, int], Optional[tuple[np.ndarray, ...]]]:
        """
        Wrapper for `stable_baselines3.common.base_class.BaseAlgorithm.predict`:

        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        if len(observations) == 0:
            return {}, state

        obs = np.asarray([observation for _, observation in observations.items()])
        action, next_state = self.ppo.predict(obs, state, episode_start, deterministic)
        return {
            agent: action[index]
            for index, (agent, _) in enumerate(observations.items())
        }, next_state

    @classmethod
    def load_trial(
        cls,
        env: ParallelEnv,
        trial_index: int | None = None,
        save_model_name: str = "dyn-ppo",
        saved_model_regex: re.Pattern | None = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ):
        """Load a model in the current directory from trial `trial_index`.
        Default to loading the last model if `trial_index` is not set."""
        saved_model_regex = saved_model_regex or re.compile(
            save_model_name + r"(\d+)\.model"
        )
        if trial_index is None:
            trial_and_model = find_last_model(saved_model_regex)
            if trial_and_model is None:
                return None
            trial_index = trial_and_model[0]
            model_file = trial_and_model[1]
        else:
            model_file = f"{save_model_name}{trial_index}.model"

        return cls.load(
            model_file,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            trial_index=trial_index,
            save_model_name=save_model_name,
            saved_model_regex=saved_model_regex,
            **kwargs,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: ParallelEnv,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        trial_index: int = 0,
        save_model_name: str = "dyn-ppo",
        saved_model_regex: re.Pattern | None = None,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = GrowableRolloutBuffer,
        **kwargs,
    ):
        """
        Wrapper of `stable_baselines3.load`:

        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        ppo_env = DummyVecEnv(
            1, env.observation_space(env.agents[0]), env.action_space(env.agents[0])
        )
        ppo = PPO.load(
            path,
            ppo_env,
            device,
            custom_objects,
            print_system_info,
            force_reset,
            rollout_buffer_class=rollout_buffer_class,
            **kwargs,
        )
        return cls(
            None,
            env,
            trial_index=trial_index,
            save_model_name=save_model_name,
            saved_model_regex=saved_model_regex,
            ppo=ppo,
        )

    def save(
        self,
        path: str | pathlib.Path | io.BufferedIOBase | None = None,
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ):
        """
        Defaults to saving to f"{self.save_model_name}{self.trial_index}.model".

        Wrapper on `stable_baselines3.PPO.save`:

        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        path = path or f"{self.save_model_name}{self.trial_index}.model"
        self.ppo.save(path, exclude, include)
        return path


def main() -> None:
    """For benchmarking."""
    from stable_baselines3.ppo import MlpPolicy

    from battlesnake_gym import BattlesnakeEnv
    from battlesnake_train.feature import (
        VIT_CLASSIFIER_NET_ARCH,
        ViTFeatureExtractor,
        vit_classifier_activation_fn,
    )

    env = BattlesnakeEnv()
    model = DynPPO.load_trial(env, save_model_name="vit-2ent-for-bench", ent_coef=0.01)
    if model is None:
        policy_kwargs = dict(
            features_extractor_class=ViTFeatureExtractor,
            net_arch=VIT_CLASSIFIER_NET_ARCH,
            activation_fn=vit_classifier_activation_fn,
        )
        model = DynPPO(
            MlpPolicy,
            env,
            save_model_name="vit-2ent-for-bench",
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

    print(f"Model trial index: {model.trial_index}.")
    model.learn_trials(1, 0x10_000, log_interval=0x100, progress_bar=True)


main() if __name__ == "__main__" else None
