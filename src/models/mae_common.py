import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Any
from .patch_embed import PatchEmbed
from .layers import Block
from .transformerpp import BlockPP
from .pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed, get_sinusoid_encoding_table
from rotary_embedding_torch import RotaryEmbedding


layernorm_wrapper = partial(nn.LayerNorm, eps=1e-6)


class Encoder(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 depth: int,
                 num_heads: int,
                 total_patches: int,
                 grid_size: Tuple[int, int],
                 mlp_ratio: float = 4.,
                 masking_mode: str = "unstructured",
                 use_cls_token: bool = False,
                 norm_layer=layernorm_wrapper,
                 use_plusplus_block: bool = False,
                 use_swiglu_final_ffn: bool = False,
                 use_rope: bool = False,
                 limit_swiglu_features: bool = False,
                 additive_swiglu: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.masking_mode = masking_mode
        self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList()
        self.use_rope = use_rope
        if limit_swiglu_features:
            print("Using limited SwiGLU features in Encoder")
        if additive_swiglu:
            print("Using additive SwiGLU in Encoder")
        for i in range(depth):
            if not use_plusplus_block:
                self.blocks.append(
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer
                    )
                )
            else:
                self.blocks.append(
                    BlockPP(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                        use_swiglu_final_ffn=use_swiglu_final_ffn,
                        use_rope=use_rope,
                        limit_swiglu_features=limit_swiglu_features,
                        additive_swiglu=additive_swiglu
                    )
                )
        self.encoder_norm = norm_layer(embed_dim)
        self.use_cls_token = use_cls_token
        self.initialize_weights()
    
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (100 - mask_ratio*100) / 100)

        if self.masking_mode == "unstructured":
            noise = torch.rand(N, L, device=x.device)
        elif self.masking_mode == "timestep":
            grid_size = self.grid_size
            noise = torch.rand(N, L//grid_size[1], device=x.device)
            noise = torch.repeat_interleave(noise, repeats=grid_size[1], dim=1)
        else:
            raise NotImplementedError(f"masking_mode={self.masking_mode} is not implemented")

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore, ids_keep
    
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size, cls_token=self.use_cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x, mask_ratio, cls_token=None, mask=None, ids_restore=None, ids_keep=None):
        if self.use_cls_token:
            x = x + self.pos_embed[:, 1:, :]
        else:
            x = x + self.pos_embed[:, :, :]
        if mask is None:
            mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        else:
            assert ids_restore is not None and ids_keep is not None
            # print("using provided mask/ids_restore/ids_keep")
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        if self.use_cls_token:
            cls_token = cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore, ids_keep

    def forward_features(self, x, cls_token=None):
        # print("in forward_encoder, after patch_embed:", x.shape)
        if self.use_cls_token:
            x = x + self.pos_embed[:, 1:, :]
        else:
            x = x + self.pos_embed[:, :, :]
        if self.use_cls_token:
            cls_token = cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        if self.use_cls_token:
            outcome = x[:, 1:, :]
        else:
            outcome = x[:, :, :]
        # print("in extract_features, x shape:", x.shape)
        
        return outcome


class Decoder(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 decoder_embed_dim: int,
                 decoder_depth: int,
                 decoder_num_heads: int,
                 total_patches: int,
                 img_patch_dim: int,
                 mlp_ratio: float = 4.,
                 use_cls_token: bool = False,
                 norm_layer=layernorm_wrapper,
                 use_plusplus_block: bool = False,
                 use_swiglu_final_ffn: bool = False,
                 use_rope: bool = False,
                 limit_swiglu_features: bool = False,
                 additive_swiglu: bool = False):
        super().__init__()
        self.total_patches = total_patches
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, total_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList()
        if limit_swiglu_features:
            print("Using limited SwiGLU features in Decoder")
        if additive_swiglu:
            print("Using additive SwiGLU in Decoder")
        for i in range(decoder_depth):
            if not use_plusplus_block:
                self.decoder_blocks.append(
                    Block(
                        dim=decoder_embed_dim,
                        num_heads=decoder_num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer
                    )
                )
            else:
                self.decoder_blocks.append(
                    BlockPP(
                        dim=decoder_embed_dim,
                        num_heads=decoder_num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                        use_swiglu_final_ffn=use_swiglu_final_ffn,
                        use_rope=use_rope,
                        limit_swiglu_features=limit_swiglu_features,
                        additive_swiglu=additive_swiglu
                    )
                )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, img_patch_dim, bias=True)
        self.use_cls_token = use_cls_token
        self.initialize_weights()

    def initialize_weights(self):
        n_patches = self.total_patches
        if self.use_cls_token:
            n_patches -= 1
        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 
                                                    n_patches, cls_token=self.use_cls_token)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=0.02)
    
    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)

        if self.use_cls_token:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat((x, mask_tokens), dim=1)
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[-1]))

        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        if self.use_cls_token:
            x = x[:, 1:, :]
        return x


encoder_configs = {
    "tiny": {
        "depth": 12, "num_heads": 3, "embed_dim": 192
    },
    "small": {
        "depth": 12, "num_heads": 6, "embed_dim": 384
    },
    "medium": {
        "depth": 12, "num_heads": 8, "embed_dim": 512
    },
    "base": {
        "depth": 12, "num_heads": 12, "embed_dim": 768
    },
    "large": {
        "depth": 24, "num_heads": 16, "embed_dim": 1024
    },
    "huge": {
        "depth": 32, "num_heads": 16, "embed_dim": 1280
    }
}