"""Agent 0, 1 are played by model 0; agent 2, 3 are played by model 1."""

from time import sleep, time

from battlesnake_gym import BattlesnakeEnv
from battlesnake_train.ppo import DynPPO

env = BattlesnakeEnv()
model0 = DynPPO.load_trial(
    env, save_model_name="dyn-ppo20prev-mlp-battlesnake", trial_index=0
)
model1 = DynPPO.load_trial(env, save_model_name="dyn-ppo-vgg16-battlesnake")
assert model0 is not None
assert model1 is not None
print(
    f"Model 0 trial index: {model0.trial_index}, model 1 trial index: {model1.trial_index}."
)

cummulative_done = {a: True for a in env.agents}
obs, _ = env.reset()

while True:  # Do not do this in IPython.
    if all(cummulative_done.values()):
        # fmt: off
        cummulative_done = {a: False for a in env.agents}; obs, _ = env.reset(); print(f"\n\n{env.render()}")
        sleep(1)
    # fmt: off
    start_time0 = time(); action0, _ = model0.predict({a: obs[a] for a in (0, 1) if a in obs}); model0_time_sec = time() - start_time0; start_time1 = time(); action1, _ = model1.predict({a: obs[a] for a in (2, 3) if a in obs}); model1_time_sec = time() - start_time1; action = action0 | action1; obs, rewards, terms, truncs, _ = env.step(action); print(f"\n\n{env.render()}\naction: {action}\ntime: {model0_time_sec}s (model0), {model1_time_sec}s (model1)\nrewards: {rewards}")
    # fmt: on
    for agent, prev_done in cummulative_done.items():
        cummulative_done[agent] = (
            prev_done or terms.get(agent, False) or truncs.get(agent, False)
        )
    sleep(max(0.0, 0.2 - (time() - start_time0)))
