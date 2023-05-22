import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,  # formerly computed from "input_tensor_"
        out_channels: int,  # formerly "filter_nb"
        kernel_size: int = 7,  # formelry "filter_size"
        stride: int = 1,  # formelry "strides"
        padding: int = 3,
        negative_slope: float = 0.1,  # formerly "alpha"
        pooling: bool = True,
        pooling_size: int = 2,
    ):  # omitting "freeze_batch" param
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=negative_slope),
            ]
        )

        if pooling:
            self.layers.append(
                nn.MaxPool2d(
                    kernel_size=pooling_size,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.conv_block = ConvBlock(**kwargs)

    def forward(self, x):
        res = self.conv_block(x)
        return x + res


# formerly "output"
class Head(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        negative_slope: float,
        dropout_rate: float,
    ):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.layers = nn.ModuleDict(
            [
                # nn.AdaptiveAvgPool2d((1, 1)),
                [
                    "linear_1",
                    nn.Linear(
                        in_features=in_features,
                        out_features=hidden_dim,
                    ),
                ],
                ["activation", nn.LeakyReLU(negative_slope)],
                ["dropout", nn.Dropout(dropout_rate)],
                [
                    "linear_2",
                    nn.Linear(
                        in_features=hidden_dim,
                        out_features=num_classes,
                    ),
                ],
                ["softmax", nn.Softmax()],
            ]
        )

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        for layer in self.layers.values():
            x = layer(x)

        return x
