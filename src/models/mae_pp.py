import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Any
from .patch_embed import PatchEmbed
from .mae import BaseMAE
from .mae_common import Encoder, Decoder, layernorm_wrapper, encoder_configs


__all__ = [
    "mae_plusplus_tiny",
    "mae_plusplus_small",
    "mae_plusplus_medium",
    "mae_plusplus_base",
    "mae_plusplus_large",
    "mae_plusplus_huge",
    "MAE_PlusPlus"
]


class MAE_PlusPlus(BaseMAE):
    def __init__(self, 
                 img_size=(200, 80), 
                 patch_size=(4, 16), 
                 in_chans=1, 
                 embed_dim=768, 
                 depth=12, num_heads=12, 
                 decoder_embed_dim=384, decoder_depth=4, 
                 decoder_num_heads=8, mlp_ratio=4, 
                 mask_ratio=0.8, masking_mode: str = "unstructured", 
                 use_cls_token: bool = True,
                 frequency_first: bool = False,
                 norm_layer=layernorm_wrapper,
                 encoder_plusplus_block: bool = False,
                 decoder_plusplus_block: bool = False,
                 encoder_use_swiglu_final_ffn: bool = False,
                 decoder_use_swiglu_final_ffn: bool = False,
                 encoder_use_rope: bool = False,
                 decoder_use_rope: bool = False,
                 limit_swiglu_features: bool = False,
                 additive_swiglu: bool = False):
        super().__init__(
            img_size=img_size, 
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            mask_ratio=mask_ratio,
            masking_mode=masking_mode,
            use_cls_token=use_cls_token,
            frequency_first=frequency_first,
            norm_layer=norm_layer)
        self.encoder = Encoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            total_patches=self.total_patches,
            grid_size=self.grid_size(),
            mlp_ratio=mlp_ratio,
            masking_mode=masking_mode,
            use_cls_token=use_cls_token,
            norm_layer=norm_layer,
            use_plusplus_block=encoder_plusplus_block,
            use_swiglu_final_ffn=encoder_use_swiglu_final_ffn,
            use_rope=encoder_use_rope,
            limit_swiglu_features=limit_swiglu_features,
            additive_swiglu=additive_swiglu
        )
        self.decoder = Decoder(
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            total_patches=self.total_patches,
            img_patch_dim=self.img_patch_dim(),
            mlp_ratio=mlp_ratio,
            use_cls_token=use_cls_token,
            norm_layer=norm_layer,
            use_plusplus_block=decoder_plusplus_block,
            use_swiglu_final_ffn=decoder_use_swiglu_final_ffn,
            use_rope=decoder_use_rope,
            limit_swiglu_features=limit_swiglu_features,
            additive_swiglu=additive_swiglu
        )
        self.initialize_weights()


def _get_mae_pp(encoder_name, **kwargs):
    img_size = kwargs.pop("img_size", (200, 80))
    patch_size = kwargs.pop("patch_size", (4, 16))
    decoder_num_heads = kwargs.pop("decoder_num_heads", 6)
    decoder_depth = kwargs.pop("decoder_depth", 4)
    decoder_embed_dim = kwargs.pop("decoder_embed_dim", 384)
    
    encoder_plusplus_block = kwargs.pop("encoder_plusplus_block", False)
    if encoder_plusplus_block:
        print("Using Transformer++ block in the encoder.")
    decoder_plusplus_block = kwargs.pop("decoder_plusplus_block", False)
    if decoder_plusplus_block:
        print("Using Transformer++ block in the decoder.")
    encoder_use_swiglu_final_ffn = kwargs.pop("encoder_use_swiglu_final_ffn", False)
    if encoder_use_swiglu_final_ffn and not encoder_plusplus_block:
        raise ValueError("SwiGLU in Encoder can only be used with Transformer++ blocks.")
    decoder_use_swiglu_final_ffn = kwargs.pop("decoder_use_swiglu_final_ffn", False)
    if decoder_use_swiglu_final_ffn and not decoder_plusplus_block:
        raise ValueError("SwiGLU in Decoder can only be used with Transformer++ blocks.")
    encoder_use_rope = kwargs.pop("encoder_use_rope", False)
    decoder_use_rope = kwargs.pop("decoder_use_rope", False)

    limit_swiglu_features = kwargs.pop("limit_swiglu_features", False)
    additive_swiglu = kwargs.pop("additive_swiglu", False)

    enc_params = encoder_configs[encoder_name]

    return MAE_PlusPlus(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=enc_params["embed_dim"],
        depth=enc_params["depth"],   
        num_heads=enc_params["num_heads"],
        decoder_num_heads=decoder_num_heads,
        decoder_depth=decoder_depth,
        decoder_embed_dim=decoder_embed_dim,
        encoder_plusplus_block=encoder_plusplus_block,
        decoder_plusplus_block=decoder_plusplus_block,
        encoder_use_swiglu_final_ffn=encoder_use_swiglu_final_ffn,
        decoder_use_swiglu_final_ffn=decoder_use_swiglu_final_ffn,
        encoder_use_rope=encoder_use_rope,
        decoder_use_rope=decoder_use_rope,
        limit_swiglu_features=limit_swiglu_features,
        additive_swiglu=additive_swiglu,
        **kwargs
    )


def mae_plusplus_tiny(**kwargs):
    return _get_mae_pp("tiny", **kwargs)


def mae_plusplus_small(**kwargs):
    return _get_mae_pp("small", **kwargs)


def mae_plusplus_medium(**kwargs):
    return _get_mae_pp("medium", **kwargs) 


def mae_plusplus_base(**kwargs):
    return _get_mae_pp("base", **kwargs)


def mae_plusplus_large(**kwargs):
    return _get_mae_pp("large", **kwargs)


def mae_plusplus_huge(**kwargs):
    return _get_mae_pp("huge", **kwargs)
