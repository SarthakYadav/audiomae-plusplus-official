import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Any
from .patch_embed import PatchEmbed
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


class BaseMAE(nn.Module):
    def __init__(self, 
                 img_size=(200, 80),
                 patch_size=(4, 16),
                 in_chans=1,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 decoder_embed_dim=384,
                 decoder_depth=4,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 mask_ratio=0.8,
                 masking_mode: str = "unstructured",
                 use_cls_token: bool = False,
                 frequency_first: bool = False,
                 norm_layer=layernorm_wrapper,
                 norm_pix_loss: bool = False,
                #  noise_position: Optional[str] = None,
                #  noise_sigma_range = (0.1, 1.),
                #  update_decoder_when_denoising: bool = False
                 ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.masking_mode = masking_mode
        self.img_size = img_size
        self.norm_pix_loss = norm_pix_loss
        self.frequency_first = frequency_first
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_chans=in_chans,
            frequency_first=frequency_first
        )
        self.num_patches = self.patch_embed.num_patches
        total_patches = self.num_patches
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            total_patches += 1
        else:
            self.cls_token = None
        self.total_patches = total_patches
        self.encoder = None
        self.decoder = None

        self.initialize_weights()
    
    def img_patch_dim(self):
        patch_size = self.patch_embed.patch_size
        return patch_size[0] * patch_size[1] * self.in_chans
    
    def patch_size(self):
        return self.patch_embed.patch_size
    
    def grid_size(self):
        return self.patch_embed.grid_size

    def initialize_weights(self):
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size(), cls_token=self.use_cls_token)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=self.use_cls_token)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.use_cls_token:
            torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size[0]*patch_size[0]*in_chans)
        """
        ph, pw = self.patch_size()
        h, w = self.grid_size()

        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, ph, w, pw))
        # standardized patch layout irrespective of whether frequency channel comes first or second
        if self.frequency_first:
            x = torch.einsum('nchpwq->nwhqpc', x)
        else:
            x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, self.img_patch_dim()))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size[0]*patch_size[0]*in_chans)
        imgs: (N, H, W, C)
        """
        ph, pw = self.patch_size()
        h, w = self.grid_size()
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * ph, w * pw))
        return imgs
    
    def forward_encoder(self, x, mask_ratio, mask=None, ids_restore=None, ids_keep=None):
        return self.encoder(x, mask_ratio, self.cls_token, mask, ids_restore, ids_keep)
    
    def forward_decoder(self, x, ids_restore):
        return self.decoder(x, ids_restore)
    
    def forward(self, imgs):
        x = self.patch_embed(imgs)
        x, mask, ids_restore, ids_keep = self.forward_encoder(x, self.mask_ratio)
        pred = self.forward_decoder(x, ids_restore)
        target = self.patchify(imgs)
        return pred, target, mask

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        outcome = self.encoder.forward_features(x, self.cls_token)
        grid_size = self.grid_size()
        if self.frequency_first:
            f, t = grid_size
        else:
            t, f = grid_size
        outcome = rearrange(outcome, 'b (f t) d -> b t (f d)', f=f, d=self.embed_dim)
        return outcome
    
    def compute_loss(self, pred, target, mask, norm_pix_loss:bool=False):
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var+1e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def set_decoder_grad(self, requires_grad: bool = True):
        for param in self.decoder.parameters():
            param.requires_grad = requires_grad


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
