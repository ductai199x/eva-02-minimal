import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .conv_layers import ConvBnAct, LayerNormOverChannels
from typing import *



class FPN(nn.Module):
    """
    Feature Pyramid Network.
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        patch_size,
        scale_factors,
        top_block=False,
        norm_layer: Optional[Callable] = None,
    ):
        """
        Args:
            in_chans (int): number of input channels.
            out_chans (int): number of output channels.
            patch_size (int): patch size of the input features.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (bool): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm_layer (nn.Module): the normalization to use.
        """
        super().__init__()
        self.top_block = top_block
        strides = [int(patch_size / scale) for scale in scale_factors]
        self.size_divisibility = strides[-1]
        dim = in_chans
        norm_layer = LayerNormOverChannels if norm_layer is None else norm_layer

        self.stages = []
        self.stage_idxs = []
        for idx, scale in enumerate(scale_factors):
            dim_ = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    norm_layer(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                dim_ = dim // 4
            elif scale == 2.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                ]
                dim_ = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            else:
                raise ValueError(f"Unknown scale {scale}")

            layers += [
                ConvBnAct(dim_, out_chans, kernel_size=1, bias=False, norm_layer=partial(norm_layer, out_chans)),
                ConvBnAct(out_chans, out_chans, kernel_size=3, padding=1, bias=False, norm_layer=partial(norm_layer, out_chans)),
            ]
            layers = nn.Sequential(*layers)

            self.stages.append(layers)
            stage_idx = int(math.log2(strides[idx]))
            self.stage_idxs.append(stage_idx)
            self.add_module(f"simfp_{stage_idx}", layers)

    def forward(self, x):
        results = {}
        for stage_idx, stage in zip(self.stage_idxs, self.stages):
            results[f"p{stage_idx}"] = stage(x)

        if self.top_block:
            results[f"p{stage_idx + 1}"] = F.max_pool2d(results[f"p{stage_idx}"], kernel_size=1, stride=2)

        return results
