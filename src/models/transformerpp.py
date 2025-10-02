import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Mlp, layernorm_wrapper, Attention, LayerScale, DropPath
from rotary_embedding_torch import RotaryEmbedding
from typing import Union


class SwiGLUFFN(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int = None,
            out_features: int = None,
            drop: float = 0.,
            norm_layer=layernorm_wrapper,
            additive=False
        ):
        super().__init__()
        if out_features is None:
            out_features = in_features
        if hidden_features is None:
            # hidden size as per "Revisiting Convolution-free Transformer for Speech Recognition", INTERSPEECH 2024
            hidden_features = int(4 * in_features * (2/3))

        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(in_features, hidden_features)
        self.drop1 = nn.Dropout(drop) if drop != 0 else nn.Identity()
        self.drop2 = nn.Dropout(drop) if drop != 0 else nn.Identity()

        self.ln = norm_layer(hidden_features)
        self.project = nn.Linear(hidden_features, out_features)
        self.additive = additive

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # PreLN has been applied to the input
        out = self.linear1(x)
        gate = self.linear2(x)

        if self.additive:
            x = F.silu(gate) + out
        else:
            x = F.silu(gate) * out    # SwiGLU

        x = self.drop1(x)
        x = self.ln(x)
        
        x = self.drop2(self.project(x))

        return x


class BlockPP(nn.Module):
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
        act_layer=nn.SiLU,
        norm_layer=layernorm_wrapper,
        mlp_layer=Mlp,
        use_swiglu_final_ffn=False,
        use_rope=False,
        limit_swiglu_features=False,
        additive_swiglu=False
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.ffn1 = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        if use_rope:
            self.rope = RotaryEmbedding(dim//num_heads)
        else:
            self.rope = None

        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop, attn_drop=attn_drop, 
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)

        if not use_swiglu_final_ffn:
            self.ffn2 = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )
        else:
            self.ffn2 = SwiGLUFFN(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio / 2) if limit_swiglu_features else None,
                out_features=dim,
                drop=proj_drop,
                norm_layer=norm_layer,
                additive=additive_swiglu
            )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + (self.ffn1(self.norm1(x)) * 0.5)
        x = x + self.drop_path1(self.ls1(self.attn(self.norm2(x), rope=self.rope)))
        x = x + self.drop_path2(self.ls2(self.ffn2(self.norm2(x))))
        return x
