import torch
import torch.nn as nn

from typing import *


class SeparableConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_features,
            in_features,
            kernel_size,
            stride,
            padding,
            groups=in_features,
            bias=bias,
        )
        self.pointwise_conv = nn.Conv2d(in_features, out_features, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class ConvBnActModule(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        stride=1,
        padding=0,
        act_layer=None,
        norm_layer=None,
        bias=True,
        use_separable_conv=False,
    ):
        super().__init__()
        if use_separable_conv and stride > 1:
            self.conv = SeparableConv2d(in_features, out_features, kernel_size, stride, padding, bias=bias)
        else:
            self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, bias=bias)
        self.norm = norm_layer() if norm_layer else nn.Identity()
        self.act = norm_layer() if act_layer else nn.Identity()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x


class ConvBnAct(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm_layer (nn.Module, optional): a normalization layer
            act_layer (nn.Module, optional): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm_layer = kwargs.pop("norm_layer", None)
        act_layer = kwargs.pop("act_layer", None)
        super().__init__(*args, **kwargs)
        self.act = act_layer() if act_layer else nn.Identity()
        self.norm = norm_layer() if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.act(self.norm(super().forward(x)))
        return x


class LayerNormOverChannels(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
