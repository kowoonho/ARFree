import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class Normalization(nn.Module):
    def __init__(self, dim, cond_dim=128, norm_type='instance', num_groups=8):
        super().__init__()
        
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm3d(dim)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(dim)
        elif norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups, dim)
        elif norm_type == 'layer3D':
            self.norm = LayerNorm(dim)
        else:
            raise ValueError(f'Invalid normalization type: {norm_type}')
        
            
    def forward(self, x):
        return self.norm(x)
    
    
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma
