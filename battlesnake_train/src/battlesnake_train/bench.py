import random

from battlesnake_gym import BattlesnakeEnv
from battlesnake_train.ppo import DynPPO
from snork_snakes_binding import SnorkTreeAgent


def benchmark(model: DynPPO, depth: int, num_games: int):
    env = model.env
    assert isinstance(env, BattlesnakeEnv)
    tree_agent = SnorkTreeAgent()
    cummulative_done = {a: False for a in env.possible_agents}
    obs, _ = env.reset()
    tree_agent_chosens = choose_tree_agents()
    n_game_played = 0
    n_game_model_won = 0
    last_done_agents: list[int] = []

    while n_game_played < num_games:
        if all(cummulative_done.values()):
            # Game over. Collect stats and reset.
            cummulative_done = {a: False for a in env.possible_agents}
            obs, _ = env.reset()
            n_game_played += 1
            if len(last_done_agents) == 1 and tree_agent_chosens[last_done_agents[0]]:
                # Model agent won.
                n_game_model_won += 1
        action, _ = model.predict(obs)
        tree_agent_action = tree_agent.step(
            env.snake_game.game_serialized(), tree_agent_chosens, depth
        )
        for agent in (2, 3):
            action[agent] = (
                env.snake_game.snake_relative_move(agent, tree_agent_action[agent]) + 1
            )
        obs, _, terms, truncs, _ = env.step(action)
        last_done_agents.clear()
        for agent, prev_done in cummulative_done.items():
            done = terms.get(agent, False) or truncs.get(agent, False)
            if done:
                last_done_agents.append(agent)
            cummulative_done[agent] = prev_done or done


def choose_tree_agents():
    tree_agent_chosens: list[bool] = []
    while len(
        tuple(chosen for chosen in tree_agent_chosens if chosen not in (1, 2, 3))
    ):
        # We want at least one of each agent to be chosen.
        tree_agent_chosens.clear()
        for _ in range(4):
            tree_agent_chosens.append(random.choice((True, False)))
    return tree_agent_chosens
