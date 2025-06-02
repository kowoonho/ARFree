import os
import math
from einops import rearrange, repeat
from functools import reduce, lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.util import import_module, tag_module, zero_init, compile_wrap, exists

from ..util import GlobalAttentionSpec, NeighborhoodAttentionSpec, LevelSpec, MappingSpec, ConsistNeighborhoodAttentionSpec


import natten


def make_HDiT_model(config):
    cfg = config.model.denoiser.model_params
    
    max_temporal_distance = config.dataset.direct.max_temporal_distance
    cf = config.dataset.train_params.cond_frames
    pf = config.dataset.train_params.pred_frames
    
    levels = []
    for depth, width, d_ff, self_attn, dropout in zip(cfg.depth_levels, cfg.width_levels, cfg.d_ff_levels, cfg.self_attn_specs, cfg.dropout_levels):
        if self_attn['type'] == 'global':
            self_attn = GlobalAttentionSpec(self_attn.get('d_head', 64))
        elif self_attn['type'] == "neighborhood":
            self_attn = NeighborhoodAttentionSpec(self_attn.get('d_head', 64), self_attn.get('kernel_size', 7))
        elif self_attn['type'] == "neighborhood_correlated":
            self_attn = ConsistNeighborhoodAttentionSpec(self_attn.get('d_head', 64), self_attn.get('kernel_size', 7))
        else:
            raise ValueError(f"Unknown attention type: {self_attn['type']}")
        
        levels.append(LevelSpec(depth, width, d_ff, self_attn, dropout))
    mapping = MappingSpec(cfg.mapping_depth, cfg.mapping_width, cfg.mapping_d_ff, cfg.mapping_dropout)
    
    denoiser_cls = import_module(f'model.module.HDiT', config.model.denoiser.type)
    
    model = denoiser_cls(
        levels=levels,
        mapping=mapping,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        motion_dim=cfg.motion_dim,
        patch_size=cfg.patch_size,
        num_classes=config.dataset.dataset_params.num_action_classes,
        mapping_cond_dim=cfg.mapping_cond_dim,
        max_temporal_distance=max_temporal_distance,
        cf=cf,
        pf=pf,
        motion_predictor_cfg=config.model.motion_predictor,
        motion_cond_type=cfg.motion_cond_type,
        cfg=cfg
    )
    
    return model

# @compile_wrap
def linear_geglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)

# @compile_wrap
def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


# @compile_wrap
def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype)**2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype)**2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


# @compile_wrap
def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = qkv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)

class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)

class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(cond_features, features, bias=False))
        # self.linear = nn.Linear(cond_features, features, bias=False)
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        if len(x.shape) == 4:
            return rms_norm(x, self.linear(cond)[:, None, None, :] + 1, self.eps)
        elif len(x.shape) == 5:
            return rms_norm(x, self.linear(cond)[:, None, None, None, :] + 1, self.eps)
    
# Rotary position embeddings

# @compile_wrap
def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


# @compile_wrap
def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos): # [B, H, W, 2]
        
        if pos.shape[-1] == 2:
            theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
            theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
            return torch.cat((theta_h, theta_w), dim=-1)
        elif pos.shape[-1] == 3:
            theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
            theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
            theta_t = pos[..., None, 2:3] * self.freqs.to(pos.dtype)
            return torch.cat((theta_h, theta_w, theta_t), dim=-1)
        else:
            raise ValueError("Unsupported position shape")

## Token merging and splitting
class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h, self.w = patch_size[0], patch_size[1]
        self.proj = nn.Linear(in_features * self.h * self.w, out_features, bias=False)

    def forward(self, x):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> b h w c')
            x = rearrange(x, 'b (h nh) (w nw) c -> b h w (nh nw c)', nh=self.h, nw=self.w)
            x = self.proj(x)
            return rearrange(x, 'b h w c -> b c h w')
        elif len(x.shape) == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, "b c t (h nh) (w nw) -> b t h w (nh nw c)", nh=self.h, nw=self.w)
            x = self.proj(x)
            x = rearrange(x, 'b t h w c -> b c t h w')
            return x
        else:
            raise ValueError("Unsupported input shape")
    
class TokenSplit(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h, self.w = patch_size[0], patch_size[1]
        self.proj = nn.Linear(in_features, out_features * self.h * self.w, bias=False)
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> b h w c')
            x = self.proj(x)
            x = rearrange(x, "b h w (nh nw c) -> b c (h nh) (w nw)", nh=self.h, nw=self.w)
            return torch.lerp(skip, x, self.fac.to(x.dtype))
        elif len(x.shape) == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, 'b c t h w -> b t h w c')
            x = self.proj(x)
            x = rearrange(x, "b t h w (nh nw c) -> b c t (h nh) (w nw)", nh=self.h, nw=self.w)
            
            return torch.lerp(skip, x, self.fac.to(x.dtype))
        else:
            raise ValueError("Unsupported input shape")
    
class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h, self.w = patch_size[0], patch_size[1]
        self.proj = nn.Linear(in_features, out_features * self.h * self.w, bias=False)

    def forward(self, x):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> b h w c')
            x = self.proj(x)
            return rearrange(x, "b h w (nh nw c) -> b c (h nh) (w nw)", nh=self.h, nw=self.w)
        elif len(x.shape) == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, 'b c t h w -> b t h w c')
            x = self.proj(x)
            return rearrange(x, "b t h w (nh nw c) -> b c t (h nh) (w nw)", nh=self.h, nw=self.w)
        else:
            raise ValueError("Unsupported input shape")
        
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))
        

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, pos, cond):
        B, C, T, H, W = x.shape
        skip = x
        x = rearrange(x, 'b c t h w -> (b t) h w c')
        cond = repeat(cond, 'b c -> (b t) c', t=T)
        x = self.norm(x, cond)
        
        qkv = self.qkv_proj(x)
        pos = rearrange(pos, "... h w e -> ... (h w) e").to(qkv.dtype)
        theta = self.pos_emb(pos)
        theta = repeat(theta, 'b s nh d -> (b t) s nh d', t=T)
        
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh (h w) e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
        theta = rearrange(theta, 'b s nh d -> b nh s d')
        q = apply_rotary_emb_(q, theta)
        k = apply_rotary_emb_(k, theta)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh (h w) e -> n h w (nh e)", h=H, w=W)
        
        x = self.dropout(x)
        x = self.out_proj(x)
        x = rearrange(x, '(b t) h w c -> b c t h w', t=T)
        return x + skip

class NeighborhoodSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, kernel_size, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.kernel_size = kernel_size
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))
        
    def extra_repr(self):
        return f"d_head={self.d_head}, kernel_size={self.kernel_size}"

    def forward(self, x, pos, cond):
        B, C, T, H, W = x.shape
        skip = x
        x = rearrange(x, 'b c t h w -> (b t) h w c')
        cond = repeat(cond, 'b c -> (b t) c', t=T)
        x = self.norm(x, cond)  # RMS Norm
        qkv = self.qkv_proj(x)
        
        q, k, v = rearrange(qkv, "b h w (m nh e) -> m b h w nh e", m=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None], 1e-6)
        theta = self.pos_emb(pos) # (b, h, w, n_heads, d_head // 4)
        theta = repeat(theta, 'b h w n d -> (b t) h w n d', t=T)
        q = apply_rotary_emb(q, theta) # (B*T, h, w, n_heads, c)
        k = apply_rotary_emb(k, theta)
        x = natten.functional.na2d(q, k, v, self.kernel_size, scale=1.0) # x: (B,64,64,2,64)
        x = rearrange(x, "n h w nh e -> n h w (nh e)")
        
        x = self.dropout(x)
        x = self.out_proj(x)
        x = rearrange(x, '(b t) h w c -> b c t h w', t=T)
        return x + skip  
        
    
class SpatioTemporalAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, cond_features, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))
        
    def forward(self, x, pos, cond):
        B, C, T, H, W = x.shape
        skip = x
        x = rearrange(x, 'b c t h w -> b h w t c')
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, 'b h w t (m nh e) -> m b h w t nh e', m=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None], 1e-6)
        theta = self.pos_emb(pos)
        q = apply_rotary_emb(q, theta)
        k = apply_rotary_emb(k, theta)
        
        q = rearrange(q, 'b h w t nh e -> b nh (h w t) e')
        k = rearrange(k, 'b h w t nh e -> b nh (h w t) e')
        v = rearrange(v, 'b h w t nh e -> b nh (h w t) e')
        
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, 'b nh (h w t) e -> b h w t (nh e)', h=H, w=W)
        
        x = self.dropout(x)
        x = self.out_proj(x)
        x = rearrange(x, 'b h w t c -> b c t h w')
        return x + skip
        

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, cond_features, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, cond_features)
        self.up_proj = LinearGEGLU(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x, cond):
        B, C, T, H, W = x.shape
        skip = x
        x = rearrange(x, 'b c t h w -> b h w t c')
        x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        x = rearrange(x, 'b h w t c -> b c t h w')
        return x + skip
   

class NeighborhoodTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        d_head,
        cond_features,
        kernel_size,
        dropout=0.0,
        cf=5,
        pf=5,
    ):
        super().__init__()
        self.cf = cf
        self.pf = pf
        
        self.spa_self_attn = NeighborhoodSelfAttentionBlock(d_model, d_head, cond_features, kernel_size, dropout=dropout)
        self.spa_ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)
        
        self.cross_frame_attn = SpatioTemporalAttentionBlock(d_model, d_head, cond_features, dropout=dropout)
        self.cross_ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)
        
    def forward(self, x, pos, cond):
        spa_pos = pos[:, :, :, 0, :2]
        
        x = self.spa_self_attn(x, spa_pos, cond)
        x = self.spa_ff(x, cond)
        
        x = self.cross_frame_attn(x, pos, cond)
        x = self.cross_ff(x, cond)
        
        return x
 

class GlobalTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_head, cond_features, dropout=0.0,
                 cf=5, pf=5):
        super().__init__()
        
        self.cf = cf
        self.pf = pf
        
        self.spa_self_attn = SelfAttentionBlock(d_model, d_head, cond_features, dropout=dropout)
        self.spa_ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)
        
        self.cross_frame_attn = SpatioTemporalAttentionBlock(d_model, d_head, cond_features, dropout=dropout)
        self.cross_ff = FeedForwardBlock(d_model, d_ff, cond_features, dropout=dropout)
        
    def forward(self, x, pos, cond):
        spa_pos = pos[:, :, :, 0, :2]
        
        x = self.spa_self_attn(x, spa_pos, cond)
        x = self.spa_ff(x, cond)
        
        x = self.cross_frame_attn(x, pos, cond)
        x = self.cross_ff(x, cond)

        return x
    
# Mapping network
class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = LinearGEGLU(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip
    
class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x
    
# Layers
class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)
    

class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)

    def forward(self, x):
        return linear_geglu(x, self.weight, self.bias)
    
    
class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x
