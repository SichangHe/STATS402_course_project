import os
import re
from threading import Thread
from time import sleep
from timeit import timeit

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from battlesnake_gym import BattlesnakeEnv, sb3_vec_env
from battlesnake_train.ppo import DynPPO

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
        if np.all(cummulative_done):
            # fmt: off
            cummulative_done = [False, False, False, False]; obs = env.reset(); print(f"{CLEAR}{env.render()}")
            sleep(1)
        # fmt: off
        action, _ = model.predict(obs); obs, rewards, done, _ = env.step(action); print(f"{CLEAR}{env.render()}\naction: {action}\nrewards: {rewards}") # type: ignore
        # fmt: on
        cummulative_done = [cummulative_done[i] or bool(done[i]) for i in range(4)]
        sleep(0.2)


def large_train():
    env = sb3_vec_env()
    last_trial_and_model = find_last_model()

    if last_trial_and_model:
        last_trial = last_trial_and_model[0]
        model = PPO.load(last_trial_and_model[1], env=env)
        print(f"Resuming training from `{last_trial_and_model[1]}`")
    else:
        last_trial = -1
        model = PPO(MlpPolicy, env)
        print("Starting training from scratch.")

    learn = lambda: model.learn(0x100_000, log_interval=0x1_000, progress_bar=True)

    for trial in range(last_trial + 1, 1000_000):
        execution_time = timeit(learn, number=1)
        print(f"Trial {trial}: took {execution_time:.2f} seconds.")
        model.save(f"ppo-mlp-battlesnake{trial}.model")


def find_last_model(regex=r"ppo-mlp-battlesnake(\d+)\.model"):
    last_trial_and_model = None
    model_regex = re.compile(regex)
    for file in os.listdir():
        match = model_regex.match(file)
        if match:
            trial = int(match.group(1))
            if last_trial_and_model is None or trial > last_trial_and_model[0]:
                last_trial_and_model = trial, file
    return last_trial_and_model


def train_dyn_ppo():
    env = BattlesnakeEnv()
    model = DynPPO(MlpPolicy, env)
    learn = lambda: model.learn(0x10_000, log_interval=0x1_000, progress_bar=True)
    execution_time = timeit(learn, number=1)
    print(f"Took {execution_time:.2f} seconds.")

    global stop_thread

    def do():
        global stop_thread
        cummulative_done = {a: True for a in env.agents}
        while not stop_thread:
            if all(cummulative_done.values()):
                # fmt: off
                cummulative_done = {a: False for a in env.agents}; obs, _ = env.reset(); print(f"{CLEAR}{env.render()}")
                sleep(1)
            # fmt: off
            action, _ = model.predict(obs); obs, rewards, terms, truncs, _ = env.step(action); print(f"{CLEAR}{env.render()}\naction: {action}\nrewards: {rewards}") # type: ignore
            # fmt: on
            for agent, prev_done in cummulative_done.items():
                cummulative_done[agent] = (
                    prev_done or terms.get(agent, False) or truncs.get(agent, False)
                )
            sleep(0.2)

    stop_thread = False
    thread = Thread(target=do)
    try:
        for _ in range(0x1_000):
            sleep(1)
    except KeyboardInterrupt:
        stop_thread = True
    finally:
        stop_thread = True
        thread.join()


def large_dyn_ppo_train():
    env = BattlesnakeEnv()
    model = DynPPO(MlpPolicy, env)
    last_trial_and_model = find_last_model(r"dyn-ppo-mlp-battlesnake(\d+)\.model")

    if last_trial_and_model:
        last_trial = last_trial_and_model[0]
        model = DynPPO.load(last_trial_and_model[1], env=env)
        print(f"Resuming training from `{last_trial_and_model[1]}`")
    else:
        last_trial = -1
        model = DynPPO(MlpPolicy, env)
        print("Starting training from scratch.")

    learn = lambda: model.learn(0x100_000, log_interval=0x1_000, progress_bar=True)

    for trial in range(last_trial + 1, 1000_000):
        execution_time = timeit(learn, number=1)
        print(f"Trial {trial}: took {execution_time:.2f} seconds.")
        model.save(f"dyn-ppo-mlp-battlesnake{trial}.model")


def hello() -> str:
    return "Hello from battlesnake-train!"
