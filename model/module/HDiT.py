
"""k-diffusion transformer diffusion models, version 2."""
from einops import rearrange, repeat
import torch
from torch import nn
from torch.nn import functional as F
import copy

from model.util import GlobalAttentionSpec, NeighborhoodAttentionSpec
from model.module.HDiT_block import TokenMerge, TokenSplit, TokenSplitWithoutSkip, \
    NeighborhoodTransformerLayer, GlobalTransformerLayer, MappingNetwork, FourierFeatures, \
        Level, RMSNorm
        
    
from model.util import tag_module, make_axial_pos, select_pos, downscale_pos, exists, import_module, \
    prob_mask_like

    
class HDiT_EDM_consistency(nn.Module):
    def __init__(self, levels, mapping, in_channels, out_channels, motion_dim, patch_size, num_classes=0, mapping_cond_dim=0,
                 max_temporal_distance=35, cf=5, pf=5, motion_predictor_cfg=None,
                 motion_cond_type=None, cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.max_temporal_distance = max_temporal_distance
        self.cf = cf
        self.pf = pf
        
        self.motion_predictor_cfg = motion_predictor_cfg
        self.motion_cond_type = motion_cond_type

        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)  # 3, 128, [4,4]

        if motion_cond_type == 'concat':
            self.motion_proj = nn.Conv3d(levels[0].width*2, levels[0].width, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)

        self.time_emb = FourierFeatures(1, mapping.width)
        self.time_in_proj = nn.Linear(mapping.width, mapping.width, bias=False)
        self.aug_emb = FourierFeatures(9, mapping.width)
        self.aug_in_proj = nn.Linear(mapping.width, mapping.width, bias=False)

        self.class_emb = nn.Embedding(num_classes, mapping.width) if num_classes else None
        self.null_cond_emb = nn.Parameter(torch.randn(1, mapping.width)) if cfg.null_cond_prob else None
        
        self.mapping_cond_in_proj = nn.Linear(mapping_cond_dim, mapping.width, bias=False) if mapping_cond_dim else None
        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping")

        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
        for i, spec in enumerate(levels):
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(spec.width, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.kernel_size, dropout=spec.dropout)
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")
            
            if i < len(levels) - 1:
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))
            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])

        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        
        self.motion_merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.motion_splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)

        # motion predictor
        if motion_dim:
            motion_predictor_class = import_module("model.module.condition", self.motion_predictor_cfg.type) if exists(self.motion_predictor_cfg) else None
            self.motion_predictor = motion_predictor_class(**self.motion_predictor_cfg.model_params, out_dim=levels[0].width) if exists(self.motion_predictor_cfg) else None
        else:
            self.motion_predictor = None
        
        nn.init.zeros_(self.patch_out.proj.weight)
        
    def forward_with_cond_scale(self, *args, cond_scale = 7., **kwargs):
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + cond_scale * (logits - null_logits)

    def forward(self, x, sigma, cond_frames, motion_cond=None, frame_indices=None, class_cond=None,
                null_cond_prob=0.):
        
        x = torch.cat([cond_frames, x], dim=2) # (B, C, T, H, W)
        
        B, C, T, H, W = x.shape
        x = self.patch_in(x)
        
        pos = make_axial_pos(x.shape[-2], x.shape[-1], self.max_temporal_distance+self.cf, device=x.device) # [H, W, max_t+1, 3]
        pos = repeat(pos, 'h w t d -> b h w t d', b=B) # [B, H, W, max_t, 3]
        
        selected_pos = select_pos(pos, frame_indices) # [B, H, W, T=10, 3]

        ## Mapping network
        if class_cond is None and self.class_emb is not None:
            raise ValueError("class_cond must be specified if num_classes > 0")

        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        aug_cond = x.new_zeros([x.shape[0], 9])
        aug_emb = self.aug_in_proj(self.aug_emb(aug_cond))
        
        if self.class_emb:
            mask = prob_mask_like((B,), null_cond_prob, device=x.device) if null_cond_prob else None
            class_emb = self.class_emb(class_cond)
            class_emb = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, class_emb) if mask is not None else class_emb
        else:
            class_emb = 0
        # mapping_emb = self.mapping_cond_in_proj(mapping_cond) if self.mapping_cond_in_proj is not None else 0
        cond = self.mapping(time_emb + class_emb + aug_emb) # [B, mapping.width]
        
       # motion prediction
        if exists(motion_cond):
            motion_pred = self.motion_predictor(motion_cond, frame_indices, class_cond)
            if self.motion_cond_type == 'concat':
                if motion_pred.shape[-1] != x.shape[-1]:
                    motion_pred = F.interpolate(motion_pred, size=(x.shape[-2],x.shape[-1]), mode='bilinear')
                motion_pred = repeat(motion_pred, 'b c h w -> b c t h w',  t=T)
                x = torch.cat([x, motion_pred], dim=1)
                x = self.motion_proj(x)

        ## Hourglass transformer
        skips, poses = [], []
        for down_level, merge, m_merge in zip(self.down_levels, self.merges, self.motion_merges):
            x = down_level(x, selected_pos, cond) # (B,64,64,128) -> (B,32,32,256)
            skips.append(x)
            poses.append(copy.deepcopy(selected_pos))
            x = merge(x)
            selected_pos = downscale_pos(selected_pos)
            
        x = self.mid_level(x, selected_pos, cond)
        
        for up_level, split, m_split, skip, selected_pos in reversed(list(zip(self.up_levels, self.splits, self.motion_splits, skips, poses))):
            x = split(x, skip)
            x = up_level(x, selected_pos, cond)
        
        ## Unpatching
        x = rearrange(x, 'b c t h w -> b t h w c')
        x = self.out_norm(x)
        x = rearrange(x, 'b t h w c -> b c t h w')
        x = self.patch_out(x)
        
        return x[:, :, self.cf:]
        