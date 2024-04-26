from stable_baselines3.common.callbacks import ProgressBarCallback
from tqdm.rich import tqdm


class DynPPOProgressBarCallback(ProgressBarCallback):
    """Corrected `stable_baselines3.common.callbacks.ProgressBarCallback`
    progress bar for `DynPPO`.

    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    pbar: tqdm

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(len(self.locals["clipped_action_map"]))
        return True

    def __del__(self):
        """Hopefully get called and prevents the bar from hanging when training
        is interrupted."""
        self.pbar.close()
