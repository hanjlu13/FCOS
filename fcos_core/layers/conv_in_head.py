import torch.nn as nn
from torch.nn import BatchNorm2d, Conv2d, Sequential


def ConvInHead(
    in_channels, out_channels, kernel_size=1, stride=1, padding=1, is_lite=False
):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    if is_lite == True:
        ReLU = nn.ReLU
        return [
            Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                groups=in_channels,
                stride=stride,
                padding=padding,
            ),
            BatchNorm2d(in_channels),
            ReLU(),
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        ]
    else:
        return [
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        ]
