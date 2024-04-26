from time import sleep, time

from stable_baselines3.ppo import MlpPolicy

from battlesnake_gym import BattlesnakeEnv
from battlesnake_train.feature import (
    VGG_CLASSIFIER_NET_ARCH,
    VGGFeatureExtractor,
    vgg_classifier_activation_fn,
)
from battlesnake_train.ppo import DynPPO

env = BattlesnakeEnv()
model = DynPPO.load_trial(env, save_model_name="dyn-ppo-vgg16-battlesnake")
if model is None:
    policy_kwargs = dict(
        features_extractor_class=VGGFeatureExtractor,
        net_arch=VGG_CLASSIFIER_NET_ARCH,
        activation_fn=vgg_classifier_activation_fn,
    )
    model = DynPPO(
        MlpPolicy,
        env,
        save_model_name="dyn-ppo-vgg16-battlesnake",
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
print(f"Model trial index: {model.trial_index}.")
model.learn_trials(100, 0x10_000, log_interval=0x100, progress_bar=True)

# Simulation.
cummulative_done = {a: True for a in env.agents}
obs, _ = env.reset()

while True:  # Do not do this in IPython.
    if all(cummulative_done.values()):
        # fmt: off
        cummulative_done = {a: False for a in env.agents}; obs, _ = env.reset(); print(f"\n\n{env.render()}")
        sleep(1)
    # fmt: off
    start_time = time(); action, _ = model.predict(obs); model_time = time() - start_time; obs, rewards, terms, truncs, _ = env.step(action); print(f"\n\n{env.render()}\naction: {action}\ntime: {model_time}sec\nrewards: {rewards}")
    # fmt: on
    for agent, prev_done in cummulative_done.items():
        cummulative_done[agent] = (
            prev_done or terms.get(agent, False) or truncs.get(agent, False)
        )
    sleep(max(0.0, 0.3 - (time() - start_time)))
