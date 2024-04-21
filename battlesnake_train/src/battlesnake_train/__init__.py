from timeit import timeit
from time import sleep

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from battlesnake_gym import make_battlesnake_env

CLEAR = "\033[2J\033[H"

def mini_train():
    env = make_battlesnake_env()
    model = PPO(MlpPolicy, env)
    learn = lambda: model.learn(1000_000, log_interval=1_000, progress_bar=True)
    execution_time = timeit(learn, number=1)
    print(f"Took {execution_time:.2f} seconds.")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    print(f"reward={mean_reward:.2f} +/- {std_reward:.2f}")

    done = [True]
    while True:
        # fmt: off
        if np.any(done):
            obs = env.reset(); print(f"{CLEAR}{env.render()}")
            sleep(1)
        action, _ = model.predict(obs); obs, _, done, _ = env.step(action); print(f"{CLEAR}{env.render()}") # type: ignore
        sleep(0.2)


def hello() -> str:
    return "Hello from battlesnake-train!"
