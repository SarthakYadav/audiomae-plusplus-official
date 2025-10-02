import math
import torch
import torch.nn as nn
import collections.abc
from itertools import repeat
from functools import partial
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Any
from timm.models.vision_transformer import LayerScale, DropPath
from torch.nn.init import _calculate_fan_in_and_fan_out


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


def _ntuple(n):
    def parse(x):
        """helper function for tuple parsing"""
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


layernorm_wrapper = partial(nn.LayerNorm, eps=1e-6)


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.
            ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_layer = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj_layer = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop != 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop != 0 else nn.Identity()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv_layer(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute((2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)
            k = rope.rotate_queries_or_keys(k)
        attn = (q @ torch.swapaxes(k, -2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = torch.swapaxes((attn @ v), 1, 2).reshape(B, N, C)
        x = self.proj_layer(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int = None,
            drop: float = 0.,
            act_layer=nn.GELU
        ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop) if drop != 0 else nn.Identity()
        self.layer2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop) if drop != 0 else nn.Identity()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.drop1(self.act1(self.layer1(x)))
        x = self.drop2(self.layer2(x))
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: int,
            qkv_bias: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values = None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=layernorm_wrapper,
            mlp_layer=Mlp

        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop, attn_drop=attn_drop, 
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class AttentionHead(nn.Module):
    def __init__(self, head_dim, attn_drop) -> None:
        super().__init__()
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop != 0. else nn.Identity()
    
    def forward(self, q, k, v, key_padding_mask=None, attn_fill_value=-65000):
        B, W, C = q.shape
        attn = (q @ torch.swapaxes(k, -2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.view(attn.shape[0], 1, W),
                attn_fill_value,
            )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        if key_padding_mask is not None:
            x.masked_fill_(key_padding_mask.unsqueeze(-1), 0.0)
        return x, attn
    
