# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from functools import partial

from ..backbone.conv_layers import ConvBnAct
from typing import List


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
class FastRCNNConvFCHead(nn.Sequential):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    def __init__(self, input_shape, *, conv_dims: List[int], fc_dims: List[int], conv_norm=None):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (dict or int): shape of the input feature (HxWxC). If int, only the channel dimension is set.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (nn.Module, optional): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape["channels"], input_shape["height"], input_shape["width"])

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = ConvBnAct(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=conv_norm is None,
                norm_layer=None if conv_norm is None else partial(conv_norm, conv_dim),
                act_layer=nn.ReLU,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        for layer in self.fcs:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    """
    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
        }
    """

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return {"channels": o}
        else:
            return {"channels": o[0], "width": o[2], "height": o[1]}
