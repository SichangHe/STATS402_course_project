from time import sleep, time

from battlesnake_gym import BattlesnakeEnv
from battlesnake_train.ppo import DynPPO
from snork_snakes_binding import SnorkTreeAgent

env = BattlesnakeEnv()
model = DynPPO.load_trial(env, save_model_name="dyn-ppo20prev-mlp-battlesnake")
print(f"Model trial index: {model.trial_index}.")
tree_agent = SnorkTreeAgent()
cummulative_done = {a: True for a in env.agents}

while True:  # Do not do this in IPython.
    if all(cummulative_done.values()):
        # fmt: off
        cummulative_done = {a: False for a in env.agents}; obs, _ = env.reset(); print(f"\n\n{env.render()}")
        sleep(1)
    start_time = time()
    action, _ = model.predict(obs)  # type: ignore[reportPossiblyUnboundVariable]
    network_time_sec = time() - start_time
    start_time = time()
    tree_agent_action = tree_agent.step(
        env.snake_game.game_serialized(), [False, False, True, True], 3
    )
    tree_time_sec = time() - start_time
    for agent in (2, 3):
        action[agent] = env.snake_game.snake_relative_move(
            agent, tree_agent_action[agent]
        ) + 1
    obs, rewards, terms, truncs, _ = env.step(action)
    print(f"\n\n{env.render()}\naction: {action}\ntree_agent_action: {tree_agent_action}\ntime: {network_time_sec}s (network), {tree_time_sec}s (tree)\nrewards: {rewards}")  # type: ignore
    # fmt: on
    for agent, prev_done in cummulative_done.items():
        cummulative_done[agent] = (
            prev_done or terms.get(agent, False) or truncs.get(agent, False)
        )
    sleep(max(0.0, 0.2 - (time() - start_time)))
