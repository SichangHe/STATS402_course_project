from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from battlesnake_gym import make_battlesnake_env


def mini_train():
    env = make_battlesnake_env()
    model = PPO(MlpPolicy, env)
    model.learn(10_000, log_interval=1_000)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    print(f"reward={mean_reward:.2f} +/- {std_reward:.2f}")


def hello() -> str:
    return "Hello from battlesnake-train!"