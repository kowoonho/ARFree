import torch
import torch.nn as nn


from model.util import noise_sampling
from model.module.condition import MotionConditioning
import random

from model.module.HDiT_block import make_HDiT_model
from model.diffusion.util import diffusion_wrapper


class ARFree_prediction(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        self.cfg = cfg
        self.diffusion_params = cfg.model.diffusion.diffusion_params
        self.noise_params = cfg.model.diffusion.noise_params
        self.motion_params = cfg.model.motion_encoder.model_params
        self.cf = cfg.dataset.train_params.cond_frames
        self.pf = cfg.dataset.train_params.pred_frames
        
        self.null_cond_prob = cfg.model.denoiser.model_params.null_cond_prob
        self.cond_scale = cfg.model.denoiser.model_params.cond_scale
        
        
        self.denoiser = make_HDiT_model(cfg)
        self.diffusion = diffusion_wrapper(self.denoiser, type=cfg.model.diffusion.type, params=self.diffusion_params, noise_cfg=self.noise_params)
        
        self.motion_encoder = MotionConditioning(**self.motion_params) if self.cfg.model.denoiser.model_params.motion_dim else None
        
    def forward(self, video_frames, frame_indices, action_classes=None):
        num_overlap_frames = random.randint(1, self.pf-1)
        cond, gt1, gt2 = video_frames[:, :, :self.cf], video_frames[:, :, self.cf:self.cf+self.pf], video_frames[:, :, self.cf+self.pf-num_overlap_frames:self.cf+self.pf*2-num_overlap_frames]
        B, C, T, H, W = gt1.shape
        
        cond = torch.cat([cond, cond], dim=0)
        gt = torch.cat([gt1, gt2], dim=0)
        
        cond_indices, gt_indices_1, gt_indices_2 = frame_indices[:, :self.cf], frame_indices[:, self.cf:self.cf+self.pf], frame_indices[:, self.cf+self.pf-num_overlap_frames:self.cf+self.pf*2-num_overlap_frames]
        frame_indices1 = torch.cat([cond_indices, gt_indices_1], dim=1)
        frame_indices2 = torch.cat([cond_indices, gt_indices_2], dim=1)
        frame_indices = torch.cat([frame_indices1, frame_indices2], dim=0)
        
        motion_cond = self.motion_encoder.global_context_encode(cond) \
            if self.cfg.model.denoiser.model_params.motion_dim else None
        
        noise = noise_sampling(shape=(B, C, T+self.pf-1, H, W), device=cond.device, noise_cfg=self.noise_params)
        noise1, noise2 = noise[:, :, :T], noise[:, :, T-num_overlap_frames:2*T-num_overlap_frames]
        noise = torch.cat([noise1, noise2], dim=0)
        
        action_classes = torch.cat([action_classes, action_classes], dim=0)
        
        loss_dict = self.diffusion.forward(cond, gt, motion_cond=motion_cond, frame_indices=frame_indices, class_cond=action_classes, noise=noise,
                                        num_overlap_frames=num_overlap_frames)
        
        loss_dict['total_loss'] = loss_dict['diffusion_loss']
        if 'overlap_consistency_loss' in loss_dict:
            loss_dict['total_loss'] += self.cfg.loss_weight.overlap_consistency_loss * loss_dict['overlap_consistency_loss']
        return loss_dict
    
    def sample_video(self, x_noise, video_frames, frame_indices, action_classes=None):
        cond, gt = video_frames[:, :, :self.cf], video_frames[:, :, self.cf:self.cf+self.pf]
        
        motion_cond = self.motion_encoder.global_context_encode(cond) \
            if self.cfg.model.denoiser.model_params.motion_dim else None
        
        sampled_video = self.diffusion.sample(
            x_noise, cond, motion_cond=motion_cond, frame_indices=frame_indices, class_cond=action_classes,
            cond_scale=self.cond_scale
        )
        
        return sampled_video
    