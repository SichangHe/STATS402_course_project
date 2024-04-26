from typing import Final

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class VGGFeatureExtractor(BaseFeaturesExtractor):
    """
    VGG16 based on `torchvision.models.vgg.vgg16`.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 4096,
    ) -> None:
        super().__init__(observation_space, features_dim)

        self.features = make_vgg_feature_net(observation_space)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.linear = nn.Sequential(
            nn.Linear(512 * 7 * 7, features_dim),
            nn.ReLU(True),
            nn.Dropout(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.features(observations)
        x = self.avgpool(x)
        x = th.flatten(x, 1)
        x = self.linear(x)
        return x


N_OUT_CHANNELS_LIST: Final[list[int | None]] = [
    64,
    64,
    None,
    128,
    128,
    None,
    256,
    256,
    256,
    None,
    512,
    512,
    512,
    None,
    512,
    512,
    512,
    None,
]


def make_vgg_feature_net(observation_space: spaces.Box):
    in_channels = observation_space.shape[0]
    layers: list[nn.Module] = []
    for out_channels in N_OUT_CHANNELS_LIST:
        if out_channels is None:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU(True))
            in_channels = out_channels
    return nn.Sequential(*layers)
