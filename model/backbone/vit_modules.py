import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .utils import window_partition, window_unpartition
from functools import partial
from typing import *


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size=(16, 16),
        stride=(16, 16),
        padding=(0, 0),
        in_chans=3,
        dim=768,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            dim (int):  dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1) # B C H W -> B H W C
        return x


class VisionRotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        patch_seq_length=16,
        feature_seq_length=None,
        base=10000,
        **kwargs,
    ):
        super().__init__()
        if feature_seq_length is None:
            feature_seq_length = patch_seq_length
        tokens = torch.arange(feature_seq_length, dtype=torch.float32) / feature_seq_length * patch_seq_length

        theta = 1.0 / (
            base ** (2 / dim * torch.arange(0, dim // 2, dtype=torch.float32)).repeat_interleave(2)
        )
        idx_theta = torch.einsum(
            "i,ij -> ij", tokens, theta.repeat(len(tokens), 1)
        )
        cos_idx_theta = idx_theta.cos()
        sin_idx_theta = idx_theta.sin()

        self.register_buffer("freqs_cos", cos_idx_theta)
        self.register_buffer("freqs_sin", sin_idx_theta)

    def forward(self, x):
        B, H, N, C = x.shape
        cos_x = x * self.freqs_cos
        sin_x = x * self.freqs_sin
        sin_x = torch.stack([-sin_x[..., 1::2], sin_x[..., ::2]], dim=-1).flatten(-2)
        return cos_x + sin_x


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        norm_layer=nn.LayerNorm,
        drop_prob=0.0,
        subln=False,
        **kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        self.drop_out = nn.Dropout(drop_prob)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop_out(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop_prob=0.0,
        use_conv=False,
        **kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _pair(bias)
        drop_prob = _pair(drop_prob)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.w1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_prob[0])
        self.w2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_prob[1])

    def forward(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.w2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=None,
        rope=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv =  nn.Linear(dim, dim * 3, bias=qkv_bias)
        nn.init.constant_(self.qkv.bias, 0)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope if rope is not None else nn.Identity()

    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        x = x.reshape(B, N, C)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 3, B, H, N, C
        q, k, v = qkv.unbind(0) # B, H, N, C


        # q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
        # k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
        # v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)
        # q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, N, C
        # k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        q = self.rope(q).type_as(v)
        k = self.rope(k).type_as(v)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
        x = x.transpose(1, 2).reshape(B, N, C)
        

        x = self.proj(x)
        x = x.reshape(B, H, W, C)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_layer=Mlp,
        window_size=0,
        rope=None,
        **kwargs,
    ):
        super().__init__()
        self.window_size = window_size

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            rope=rope,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_prob=proj_drop,
            subln=True if mlp_layer == SwiGLU else False,
        )

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            x, pad_h, pad_w = window_partition(x, self.window_size)
            x = self.attn(x)
            x = window_unpartition(x, self.window_size, pad_h, pad_w, H, W)
        else:
            x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.reshape(B, H, W, C)
        return x


