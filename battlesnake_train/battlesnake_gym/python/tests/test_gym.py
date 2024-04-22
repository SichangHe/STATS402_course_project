import warnings

import numpy as np
import supersuit as ss
from gymnasium import Env, spaces
from pettingzoo.test import parallel_api_test
from stable_baselines3.common.env_checker import check_for_nested_spaces  # type: ignore
from stable_baselines3.common.env_checker import (
    _check_box_obs,
    _check_goal_env_compute_reward,
    _check_goal_env_obs,
    _check_nan,
    _check_render,
    _check_spaces,
    _check_unsupported_spaces,
    _is_goal_env,
    _is_numpy_array_space,
)
from supersuit.vector import ConcatVecEnv
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from battlesnake_gym import BattlesnakeEnv, gymnasium_vector_env


def test_parallel_api() -> None:
    env = BattlesnakeEnv()
    parallel_api_test(env, num_cycles=1_000)


def test_gymnasium_vector_env() -> None:
    env = gymnasium_vector_env()
    check_vec_env(env, skip_render_check=False)


def test_sb3_env() -> None:
    battlesnake_env = BattlesnakeEnv()
    black_death_env: black_death_par = ss.black_death_v3(battlesnake_env)  # type: ignore
    markov_vector_env = ss.pettingzoo_env_to_vec_env_v1(black_death_env)
    sb3_vec_env_wrapper = ss.concat_vec_envs_v1(
        markov_vector_env, 4, base_class="stable_baselines3"
    )
    assert isinstance(sb3_vec_env_wrapper, SB3VecEnvWrapper)
    env = sb3_vec_env_wrapper.venv
    assert isinstance(env, ConcatVecEnv), type(env)
    check_vec_env(env, skip_render_check=False)


def test_no_vec_env_wrapper() -> None:
    battlesnake_env = BattlesnakeEnv()
    black_death_env: black_death_par = ss.black_death_v3(battlesnake_env)  # type: ignore
    markov_vector_env = ss.pettingzoo_env_to_vec_env_v1(black_death_env)
    check_vec_env(markov_vector_env, skip_render_check=False)


def test_no_black_death() -> None:
    battlesnake_env = BattlesnakeEnv()
    markov_vector_env = ss.pettingzoo_env_to_vec_env_v1(battlesnake_env)
    sb3_vec_env_wrapper = ss.concat_vec_envs_v1(
        markov_vector_env, 4, base_class="stable_baselines3"
    )
    assert isinstance(sb3_vec_env_wrapper, SB3VecEnvWrapper)
    env = sb3_vec_env_wrapper.venv
    assert isinstance(env, ConcatVecEnv), type(env)
    check_vec_env(env, skip_render_check=False)


def check_vec_env(
    env,
    warn: bool = True,
    skip_render_check: bool = True,
    skip_gym_inheritance_check: bool = False,
) -> None:
    """
    Copied from `stable_baselines3/common/env_checker.py`.

    Check that an environment follows Gym API.
    This is particularly useful when using a custom environment.
    Please take a look at https://gymnasium.farama.org/api/env/
    for more information about the API.

    It also optionally check that the environment is compatible with Stable-Baselines.

    :param env: The Gym environment that will be checked
    :param warn: Whether to output additional warnings
        mainly related to the interaction with Stable Baselines
    :param skip_render_check: Whether to skip the checks for the render method.
        True by default (useful for the CI)
    """
    if not skip_gym_inheritance_check:
        assert isinstance(
            env, Env
        ), "Your environment must inherit from the gymnasium.Env class cf. https://gymnasium.farama.org/api/env/"

    # ============= Check the spaces (observation and action) ================
    _check_spaces(env)

    # Define aliases for convenience
    observation_space = env.observation_space
    action_space = env.action_space

    try:
        env.reset(seed=0)
    except TypeError as e:
        raise TypeError("The reset() method must accept a `seed` parameter") from e

    # Warn the user if needed.
    # A warning means that the environment may run but not work properly with Stable Baselines algorithms
    if warn:
        _check_unsupported_spaces(env, observation_space, action_space)

        obs_spaces = (
            observation_space.spaces
            if isinstance(observation_space, spaces.Dict)
            else {"": observation_space}
        )
        for key, space in obs_spaces.items():
            if isinstance(space, spaces.Box):
                _check_box_obs(space, key)

        # Check for the action space, it may lead to hard-to-debug issues
        if isinstance(action_space, spaces.Box) and (
            np.any(np.abs(action_space.low) != np.abs(action_space.high))
            or np.any(action_space.low != -1)
            or np.any(action_space.high != 1)
        ):
            warnings.warn(
                "We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) "
                "cf. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html"
            )

        if isinstance(action_space, spaces.Box):
            assert np.all(
                np.isfinite(np.array([action_space.low, action_space.high]))
            ), "Continuous action space must have a finite lower and upper bound"

        if isinstance(action_space, spaces.Box) and action_space.dtype != np.dtype(
            np.float32
        ):
            warnings.warn(
                f"Your action space has dtype {action_space.dtype}, we recommend using np.float32 to avoid cast errors."
            )

    # If Sequence observation space, do not check the observation any further
    if isinstance(observation_space, spaces.Sequence):
        return

    # ============ Check the returned values ===============
    _check_returned_values(env, observation_space, action_space)

    # ==== Check the render method and the declared render modes ====
    if not skip_render_check:
        _check_render(env, warn)  # pragma: no cover

    try:
        check_for_nested_spaces(env.observation_space)

        # The check doesn't support nested observations/dict actions
        # A warning about it has already been emitted
        class FirstElem(Env):
            observation_space = env.observation_space
            action_space = env.action_space

            def reset(self, seed=None):  # type: ignore
                observations, infos = env.reset(seed)  # type: ignore
                return observations[0], infos[0]

        _check_nan(FirstElem())
    except NotImplementedError:
        pass


def _check_returned_values(
    env, observation_space: spaces.Space, action_space: spaces.Space
) -> None:
    """
    Copied from `stable_baselines3/common/env_checker.py`.

    Check the returned values by the env when calling `.reset()` or `.step()` methods.
    """
    # because env inherits from gymnasium.Env, we assume that `reset()` and `step()` methods exists
    reset_returns = env.reset()
    assert isinstance(reset_returns, tuple), "`reset()` must return a tuple (obs, info)"
    assert (
        len(reset_returns) == 2
    ), f"`reset()` must return a tuple of size 2 (obs, info), not {len(reset_returns)}"
    obs, info = reset_returns
    # assert isinstance(info, dict), f"The second element of the tuple return by `reset()` must be a dictionary not {info}"

    if _is_goal_env(env):
        # Make mypy happy, already checked
        assert isinstance(observation_space, spaces.Dict)
        _check_goal_env_obs(obs, observation_space, "reset")
    elif isinstance(observation_space, spaces.Dict):
        assert isinstance(
            obs, dict
        ), "The observation returned by `reset()` must be a dictionary"

        if not obs.keys() == observation_space.spaces.keys():
            raise AssertionError(
                "The observation keys returned by `reset()` must match the observation "
                f"space keys: {obs.keys()} != {observation_space.spaces.keys()}"
            )

        for key in observation_space.spaces.keys():
            try:
                _check_obs(obs[key], observation_space.spaces[key], "reset")
            except AssertionError as e:
                raise AssertionError(
                    f"Error while checking key={key}: " + str(e)
                ) from e
    else:
        _check_obs(obs, observation_space, "reset")

    # Sample a random action
    action = np.asarray([action_space.sample() for _ in range(32)])
    data = env.step(action)

    assert len(data) == 5, (
        "The `step()` method must return five values: "
        f"obs, reward, terminated, truncated, info. Actual: {len(data)} values returned."
    )

    # Unpack
    obs, reward, terminated, truncated, info = data

    if isinstance(observation_space, spaces.Dict):
        assert isinstance(
            obs, dict
        ), "The observation returned by `step()` must be a dictionary"

        # Additional checks for GoalEnvs
        if _is_goal_env(env):
            # Make mypy happy, already checked
            assert isinstance(observation_space, spaces.Dict)
            _check_goal_env_obs(obs, observation_space, "step")
            _check_goal_env_compute_reward(obs, env, float(reward), info)

        if not obs.keys() == observation_space.spaces.keys():
            raise AssertionError(
                "The observation keys returned by `step()` must match the observation "
                f"space keys: {obs.keys()} != {observation_space.spaces.keys()}"
            )

        for key in observation_space.spaces.keys():
            try:
                _check_obs(obs[key], observation_space.spaces[key], "step")
            except AssertionError as e:
                raise AssertionError(
                    f"Error while checking key={key}: " + str(e)
                ) from e

    else:
        _check_obs(obs, observation_space, "step")

    # We also allow int because the reward will be cast to float
    assert isinstance(
        reward[0], (float, int, np.floating)
    ), "The reward returned by `step()` must be a float"
    assert isinstance(
        terminated[0], (bool, np.unsignedinteger)
    ), "The `terminated` signal must be a boolean"
    assert isinstance(
        truncated[0], (bool, np.unsignedinteger)
    ), "The `truncated` signal must be a boolean"
    assert isinstance(
        info[0], dict
    ), "The `info` returned by `step()` must be a python dictionary"

    # Goal conditioned env
    if _is_goal_env(env):
        # for mypy, env.unwrapped was checked by _is_goal_env()
        assert hasattr(env, "compute_reward")
        assert reward == env.compute_reward(
            obs["achieved_goal"], obs["desired_goal"], info
        )


def _check_obs(
    obs: tuple | dict | np.ndarray | int,
    observation_space: spaces.Space,
    method_name: str,
) -> None:
    """
    Copied from `stable_baselines3/common/env_checker.py`.

    Check that the observation returned by the environment
    correspond to the declared one.
    """
    if not isinstance(observation_space, spaces.Tuple):
        assert not isinstance(
            obs, tuple
        ), f"The observation returned by the `{method_name}()` method should be a single value, not a tuple"

    # The check for a GoalEnv is done by the base class
    if isinstance(observation_space, spaces.Discrete):
        # Since https://github.com/Farama-Foundation/Gymnasium/pull/141,
        # `sample()` will return a np.int64 instead of an int
        assert np.issubdtype(
            type(obs), np.integer
        ), f"The observation returned by `{method_name}()` method must be an int"
    elif _is_numpy_array_space(observation_space):
        assert isinstance(
            obs, np.ndarray
        ), f"The observation returned by `{method_name}()` method must be a numpy array"

    # Additional checks for numpy arrays, so the error message is clearer (see GH#1399)
    if isinstance(obs, np.ndarray):
        # check obs dimensions, dtype and bounds
        assert observation_space.shape == obs.shape[1:], (
            f"The observation returned by the `{method_name}()` method does not match the shape "
            f"of the given observation space {observation_space}. "
            f"Expected: {observation_space.shape}, actual shape: {obs.shape}"
        )
        assert np.can_cast(obs.dtype, observation_space.dtype), (
            f"The observation returned by the `{method_name}()` method does not match the data type (cannot cast) "
            f"of the given observation space {observation_space}. "
            f"Expected: {observation_space.dtype}, actual dtype: {obs.dtype}"
        )
        if isinstance(observation_space, spaces.Box):
            lower_bounds, upper_bounds = observation_space.low, observation_space.high
            # Expose all invalid indices at once
            invalid_indices = np.where(
                np.logical_or(obs < lower_bounds, obs > upper_bounds)
            )
            if (obs > upper_bounds).any() or (obs < lower_bounds).any():
                message = (
                    f"The observation returned by the `{method_name}()` method does not match the bounds "
                    f"of the given observation space {observation_space}. \n"
                )
                message += f"{len(invalid_indices[0])} invalid indices: \n"

                for index in zip(*invalid_indices):
                    index_str = ",".join(map(str, index))
                    message += (
                        f"Expected: {lower_bounds[index]} <= obs[{index_str}] <= {upper_bounds[index]}, "
                        f"actual value: {obs[index]} \n"
                    )

                raise AssertionError(message)

    assert observation_space.contains(obs[0]), (  # type: ignore
        f"The observation returned by the `{method_name}()` method "
        f"does not match the given observation space {observation_space}"
    )
