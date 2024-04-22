from time import sleep
from timeit import timeit

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from battlesnake_gym import sb3_vec_env

CLEAR = "\033[2J\033[H"


def mini_train():
    env = sb3_vec_env()
    model = PPO(MlpPolicy, env)
    learn = lambda: model.learn(0x100_000, log_interval=0x1_000, progress_bar=True)
    execution_time = timeit(learn, number=1)
    print(f"Took {execution_time:.2f} seconds.")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    print(f"reward={mean_reward:.2f} +/- {std_reward:.2f}")

    cummulative_done = [True, True, True, True]
    while True:
        # fmt: off
        if np.all(cummulative_done):
            cummulative_done = [False, False, False, False]; 
            obs = env.reset(); print(f"{CLEAR}{env.render()}")
            sleep(1)
        action, _ = model.predict(obs); obs, _, done, _ = env.step(action); print(f"{CLEAR}{env.render()}") # type: ignore
        cummulative_done = [cummulative_done[i] or bool(done[i]) for i in range(4) ]
        sleep(0.2)


def large_train():
    env = sb3_vec_env()
    model = PPO(MlpPolicy, env)
    learn = lambda: model.learn(0x100_000, log_interval=0x1_000, progress_bar=True)
    for trial in range(1000_000):
        execution_time = timeit(learn, number=1)
        print(f"Trial {trial}: took {execution_time:.2f} seconds.")
        model.save(f"ppo-mlp-battlesnake{trial}.model")


def hello() -> str:
    return "Hello from battlesnake-train!"
