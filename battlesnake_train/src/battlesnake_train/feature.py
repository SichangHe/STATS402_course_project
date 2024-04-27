from typing import Final

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class VGGFeatureExtractor(BaseFeaturesExtractor):
    """
    Based on `torchvision.models.vgg.vgg16`, with 2 fewer max-pool layers and
    an additional residual layer.

    :param observation_space:
    :param features_dim: *Ignored*. Fixed at 1024.
        Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param in_width: Width of the input image.
    :param residual_layer_width: Width of the residual layer.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 1024,
        in_width: int = 21,
        residual_layer_width: int = 2048,
    ) -> None:
        features_dim = 1024
        in_channels = observation_space.shape[0]  # 9
        super().__init__(observation_space, features_dim)

        self.features = make_vgg_feature_net(in_channels)
        feature_width = vgg_out_width(in_width)  # 2
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.residual = nn.Sequential(
            nn.Flatten(1),  # -> (batch_size, 3969)
            nn.Linear(
                in_channels * in_width * in_width,  # 3969
                residual_layer_width,
            ),
            nn.Linear(
                residual_layer_width,
                256 * feature_width * feature_width,  # 1024
            ),
            nn.ReLU(True),
            nn.Dropout(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations: (batch_size, 9, 21, 21)
        x = self.features(observations)  # (batch_size, 256, 2, 2)
        x = self.avgpool(x)
        x = th.flatten(x, 1)  # (batch_size, 256 * 2 * 2)
        x += self.residual(observations)
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
    # 512, # These layers are disabled because our width is 21 not 224.
    # 512,
    # 512,
    # None,
    # 512,
    # 512,
    # 512,
    # None,
]


def make_vgg_feature_net(in_channels: int):
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


def vgg_out_width(in_width: int):
    out_width = in_width
    for out_channels in N_OUT_CHANNELS_LIST:
        if out_channels is None:
            out_width //= 2
    return out_width


VGG_CLASSIFIER_NET_ARCH: Final = [256, 256]


def vgg_classifier_activation_fn():
    return nn.Sequential(
        nn.ReLU(True),
        nn.Dropout(),
    )
