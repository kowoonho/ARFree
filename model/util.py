import os
import math
import torch
import random
import numpy as np
from einops import repeat, rearrange
from inspect import isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import update_wrapper
from dataclasses import dataclass
from typing import Union

def import_module(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def temporal_swap(x, fn, **kwargs):
    x = rearrange(x, 'b (k c) f h w -> b c (k f) h w', k=2)
    x = fn(x, **kwargs)
    x = rearrange(x, 'b c (k f) h w -> b (k c) f h w', k=2)
    return x

def divide_and_concat(x, fn, **kwargs):
    x1, x2 = torch.chunk(x, 2, dim=2)
    
    x1 = fn(x1, **kwargs)
    x2 = fn(x2, **kwargs)
    
    return torch.cat((x1, x2), dim=2)

def noise_sampling(shape, device, noise_cfg=None):
    b, c, f, h, w = shape
    
    if noise_cfg.noise_sampling_method == 'vanilla':
        noise = torch.randn(shape, device=device)
    elif noise_cfg.noise_sampling_method == 'pyoco_mixed':
        noise_alpha_squared = float(noise_cfg.noise_alpha) ** 2
        shared_noise = torch.randn((b, c, 1, h, w), device=device) * math.sqrt((noise_alpha_squared) / (1 + noise_alpha_squared))
        ind_noise = torch.randn(shape, device=device) * math.sqrt(1 / (1 + noise_alpha_squared))
        noise = shared_noise + ind_noise
    elif noise_cfg.noise_sampling_method == 'pyoco_progressive':
        noise_alpha_squared = float(noise_cfg.noise_alpha) ** 2
        noise = torch.randn(shape, device=device)
        ind_noise = torch.randn(shape, device=device) * math.sqrt(1 / (1 + noise_alpha_squared))
        for i in range(1, noise.shape[2]):
            noise[:, :, i, :, :] = noise[:, :, i - 1, :, :] * math.sqrt((noise_alpha_squared) / (1 + noise_alpha_squared)) + ind_noise[:, :, i, :, :]
    elif noise_cfg.noise_sampling_method == 'fixed_noise':
        img_shape = (b, c, h, w)
        noise = repeat(torch.randn(img_shape, device=device), 'b c h w -> b c f h w', f=f)
    else:
        raise ValueError(f"Unknown noise sampling method {noise_cfg.noise_sampling_method}")

    return noise

def make_grid(h_pos, w_pos, t_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, t_pos, indexing='ij'), dim=-1)
    h, w, t, d = grid.shape
    return grid

def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def make_axial_pos(h, w, t, dtype=None, device=None):
    h_pos = centers(-1, 1, h, dtype=dtype, device=device)
    w_pos = centers(-1, 1, w, dtype=dtype, device=device)
    t_pos = centers(0, 1, t, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos, t_pos)

def select_pos(pos, frame_indices):
    B, H, W,_, C = pos.shape
    T = frame_indices.shape[-1]
    expanded_frame_indices = frame_indices[:, None, None, :, None].expand(B, H, W, T, C)
    selected_position = torch.gather(pos, 3, expanded_frame_indices)
    return selected_position

def downscale_pos(pos):
    pos = rearrange(pos, 'b h w t d -> b t h w d')
    pos = rearrange(pos, 'b t (h nh) (w nw) d -> b t h w (nh nw) d', nh=2, nw=2)
    down_pos = torch.mean(pos, dim=-2)
    down_pos = rearrange(down_pos, 'b t h w d -> b h w t d')
    
    return down_pos
    

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

def filter_params(function, module):
    for param in module.parameters():
        tags = getattr(param, "_tags", set())
        if function(tags):
            yield param

def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param

def tag_module(module, tag):
    for param in module.parameters():
        tag_param(param, tag)
    return module

def get_use_compile():
    return os.environ.get("USE_COMPILE", "1") == "1"

# Kernels
class compile_wrap:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._compiled_function = None
        update_wrapper(self, function)

    @property
    def compiled_function(self):
        if self._compiled_function is not None:
            return self._compiled_function
        if get_use_compile():
            torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
            torch._dynamo.config.suppress_errors = True
            try:
                self._compiled_function = torch.compile(self.function, *self.args, **self.kwargs)
            except RuntimeError:
                self._compiled_function = self.function
        else:
            self._compiled_function = self.function
        return self._compiled_function

    def __call__(self, *args, **kwargs):
        return self.compiled_function(*args, **kwargs)

# Configuration

@dataclass
class GlobalAttentionSpec:
    d_head: int

@dataclass
class NeighborhoodAttentionSpec:
    d_head: int
    kernel_size: int

@dataclass
class ConsistNeighborhoodAttentionSpec:
    d_head: int
    kernel_size: int


@dataclass
class LevelSpec:
    depth: int
    width: int
    d_ff: int
    self_attn: Union[GlobalAttentionSpec, NeighborhoodAttentionSpec, ConsistNeighborhoodAttentionSpec]
    dropout: float


@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob
    
def frame_idx_to_temp_dist(frame_indices, num_cond_frames):
    cond_frame_idx, pred_frame_idx = frame_indices[:, :num_cond_frames], frame_indices[:, num_cond_frames:]
    temp_dist = pred_frame_idx[:, 0] - cond_frame_idx[:, 0]
    
    return temp_dist 

def compute_overlap_feature(x1, x2, num_overlap_frames):
    overlap1 = x1[:, :, -num_overlap_frames:]
    overlap2 = x2[:, :, :num_overlap_frames]
    
    average_overlap = (overlap1 + overlap2) / 2
    
    non_overlap1 = x1[:, :, :-num_overlap_frames]
    non_overlap2 = x2[:, :, num_overlap_frames:]
    
    overlap_feat = torch.cat([non_overlap1, average_overlap, non_overlap2], dim=2)
    return overlap_feat
