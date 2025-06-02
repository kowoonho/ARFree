from math import sqrt
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, repeat, reduce
from model.util import exists, default, noise_sampling, append_dims
from datasets.dataset import normalize_img, unnormalize_img
from model.diffusion.util import make_sample_density, get_sigmas_karras

from tqdm.auto import trange
from scipy import integrate

class EDM_karras(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(
        self,
        model,
        *,
        image_size,
        num_frames,
        channels = 3,
        num_sample_steps = 50, # number of sampling steps
        sigma_min = 1e-2,     # min noise level
        sigma_max = 160,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7.,               # controls the sampling schedule
        weighting='karras',
        scales=1,
        noise_cfg=None,
        sampler_type='lms',
        overlap_consistency_loss=False,
    ):
        super().__init__()
        self.model = model
        self.sample_density = make_sample_density(
            sigma_data, sigma_min, sigma_max, image_size
        )
        
        self.sample_sigmas = get_sigmas_karras(num_sample_steps, sigma_min, sigma_max, rho)
        
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.scales = scales
        self.num_sample_steps = num_sample_steps
        self.noise_cfg = noise_cfg
        self.sampler_type = sampler_type
        self.overlap_consistency_loss = overlap_consistency_loss
            
        if callable(weighting):
            self.weighting = weighting
        if weighting == 'karras':
            self.weighting = torch.ones_like
        elif weighting == 'soft-min-snr':
            self.weighting = self._weighting_soft_min_snr
        elif weighting == 'snr':
            self.weighting = self._weighting_snr
        else:
            raise ValueError(f'Unknown weighting type {weighting}')

    def _weighting_soft_min_snr(self, sigma):
        return (sigma * self.sigma_data) ** 2 / (sigma ** 2 + self.sigma_data ** 2) ** 2

    def _weighting_snr(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in
    
    def to_d(self, x, sigma, denoised):
        return (x - denoised) / sigma
    
    def linear_multistep_coeff(self, order, t, i, j):
        if order - 1 > i:
            raise ValueError(f'Order {order} too high for step {i}')
        def fn(tau):
            prod = 1.
            for k in range(order):
                if j == k:
                    continue
                prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
            return prod
        return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]
  
    
    @torch.no_grad()
    def sample(self, x_T, cond_frames, motion_cond, frame_indices, class_cond, cond_scale=1.,):
        if self.sampler_type == 'lms':
            return self.sample_lms(x_T, cond_frames, motion_cond, frame_indices, class_cond, cond_scale)
        elif self.sampler_type == 'dpmpp_3m_sde':
            return self.sample_dpmpp_3m_sde(x_T, cond_frames, motion_cond, frame_indices, class_cond, cond_scale)
            
    
    @torch.no_grad()
    def sample_lms(self, x_T, cond_frames, motion_cond, frame_indices, class_cond, cond_scale=1., order=4):
        sigmas = self.sample_sigmas
        
        x_T = x_T * self.sigma_max
        s_in = x_T.new_ones(x_T.shape[0])
        sigmas_cpu = sigmas.detach().cpu().numpy()
        
        ds = []
        x = x_T
        for i in trange(len(sigmas) - 1, disable=False):
            denoised = self.denoise(x, sigmas[i] * s_in, cond_frames, motion_cond, frame_indices, class_cond, cond_scale=cond_scale)
            d = self.to_d(x, sigmas[i], denoised)
            ds.append(d)
            if len(ds) > order:
                ds.pop(0)
            cur_order = min(i+1, order)
            coeffs = [self.linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
            
        return unnormalize_img(x)

    def forward(self, x_start_cond, x_start_pred, motion_cond, frame_indices, class_cond=None, noise=None,
                       num_overlap_frames=0, null_cond_prob=0.):
        sigma = self.sample_density([x_start_cond.shape[0]//2], device=x_start_cond.device)
        sigma = torch.cat([sigma, sigma], dim=0)
        
        c_skip, c_out, c_in = [append_dims(x, x_start_pred.ndim) for x in self.get_scalings(sigma)]
        c_weight = self.weighting(sigma)
        
        noised_input = x_start_pred + noise * append_dims(sigma, x_start_pred.ndim)
        
        model_out = self.model(
            noised_input * c_in,
            sigma=sigma,
            cond_frames=x_start_cond,
            motion_cond=motion_cond,
            frame_indices=frame_indices,
            class_cond=class_cond,
            null_cond_prob=null_cond_prob,
        )
        loss_dict = {}
        
        if self.overlap_consistency_loss:
            model_out1, model_out2 = model_out.chunk(2, dim=0)
            overlap_consistency_loss = F.mse_loss(model_out1[:, :, -num_overlap_frames:], model_out2[:, :, :num_overlap_frames])
            loss_dict['overlap_consistency_loss'] = overlap_consistency_loss
            
        
        target = (x_start_pred - c_skip * noised_input) / c_out
        
        loss = ((model_out - target) ** 2).flatten(1).mean(1) * c_weight
        
        loss_dict['diffusion_loss'] = loss.mean()
        
        return loss_dict
    
    def denoise(self, x, sigma, cond_frames, motion_cond, frame_indices, class_cond=None, cond_scale=1., noise=None):    
        
        c_skip, c_out, c_in = [append_dims(s, x.ndim) for s in self.get_scalings(sigma)]
        out = self.model.forward_with_cond_scale(
            x * c_in,
            sigma=sigma,
            cond_frames=cond_frames,
            motion_cond=motion_cond,
            frame_indices=frame_indices,
            class_cond=class_cond,
            cond_scale=cond_scale,
        )
        return out * c_out + x * c_skip
    

  