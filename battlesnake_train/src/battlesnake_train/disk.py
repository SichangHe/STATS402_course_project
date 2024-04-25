import os
import re
from typing import Iterable

def find_last_model(regex: str | re.Pattern = r"ppo-mlp-battlesnake(\d+)\.model"):
    last_trial_and_model = None
    model_regex = re.compile(regex) if isinstance(regex, str) else regex
    for trial, file in all_prev_models(model_regex):
        if last_trial_and_model is None or trial > last_trial_and_model[0]:
            last_trial_and_model = trial, file
    return last_trial_and_model


def all_prev_models(regex: re.Pattern) -> Iterable[tuple[int, str]]:
    """All previous models in current directory."""
    for file in os.listdir():
        match = regex.match(file)
        if match:
            trial = int(match.group(1))
            yield (trial, file)
