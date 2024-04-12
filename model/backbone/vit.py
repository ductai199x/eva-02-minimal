import torch
import torch.nn as nn
from functools import partial

from .fpn import FPN
from .utils import get_abs_pos
from .vit_modules import *
from typing import *


class ViT(nn.Module):
    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4 * 2 / 3,
        qkv_bias=True,
        qk_norm=False,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer: Optional[Callable] = None,
        act_layer: Optional[Callable] = None,
        block_fn: Callable = Block,
        mlp_layer: Callable = SwiGLU,
        patch_hw_seq_length=16,
        interpolate_freq=True,
        window_size=0,
        window_block_idxs=(),
        use_abs_pos=True,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        **kwargs,
    ):
        """
        Args:
            init_patch_size: Initial patch size.
            n_patch_hw: Number of patches in height and width.
            in_chans: Number of image input channels.
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            qk_norm: Enable normalization for qk if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            pre_norm: Enable pre-normalization if True.
            proj_drop_rate: Projection dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
            mlp_layer: MLP layer.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        hw_seq_len = img_size // patch_size
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        self.rope_win = VisionRotaryPositionalEmbedding(
            dim=embed_dim // num_heads,
            patch_seq_length=patch_hw_seq_length,
            feature_seq_length=window_size ** 2 if interpolate_freq else None,
        )
        self.rope_glb = VisionRotaryPositionalEmbedding(
            dim=embed_dim // num_heads,
            patch_seq_length=patch_hw_seq_length,
            feature_seq_length=hw_seq_len ** 2 if interpolate_freq else None,
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    window_size=window_size if i in window_block_idxs else 0,
                    rope=self.rope_win if i in window_block_idxs else self.rope_glb,
                )
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)  # x is now BHWC
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2]))


        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # x is now BCHW
        return x


class ViTDetBackBone(nn.Module):
    def __init__(
        self,
        vit_config,
        fpn_config,
    ):
        super().__init__()
        self.vit = ViT(**vit_config)
        self.fpn = FPN(**fpn_config)
        self.size_divisibility = self.fpn.size_divisibility
        self.square_size = vit_config["img_size"]
        self.img_size = self.vit.img_size
        self.patch_size = self.vit.patch_size
        self.padding_constraints = {
            "size_divisibility": self.size_divisibility,
            "square_size": self.square_size,
        }

    def forward(self, x):
        x = self.vit(x)
        x = self.fpn(x)
        return x
