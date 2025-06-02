import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from model.util import zero_init
from model.module.attention import AttentionLayer, SinusoidalPosEmb, CrossAttentionLayer
import math

from datasets.dataset import unnormalize_img, normalize_img

class SinusoidalPosEmb3D(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta= theta

    def forward(self, pts):
        x = pts[..., 0]
        y = pts[..., 1]
        z = pts[..., 2]

        device = pts.device
        div_dim = self.dim // 6

        x_emb = math.log(self.theta) / (div_dim - 1)
        x_emb = torch.exp(torch.arange(div_dim, device=device) * -x_emb)
        x_emb = x[:, None] * x_emb[None, :]
        x_emb = torch.cat((x_emb.sin(), x_emb.cos()), dim=-1)

        y_emb = math.log(self.theta) / (div_dim - 1)
        y_emb = torch.exp(torch.arange(div_dim, device=device) * -y_emb)
        y_emb = y[:, None] * y_emb[None, :]
        y_emb = torch.cat((y_emb.sin(), y_emb.cos()), dim=-1)

        z_emb = math.log(self.theta) / (div_dim - 1)
        z_emb = torch.exp(torch.arange(div_dim, device=device) * -z_emb)
        z_emb = z[:, None] * z_emb[None, :]
        z_emb = torch.cat((z_emb.sin(), z_emb.cos()), dim=-1)

        emb = torch.cat((x_emb, y_emb, z_emb), dim=-1)
        return emb


class PositionalEmbedding3D(nn.Module):
    def __init__(
            self,
            shape,
            pos_dim=60,
            out_channels=64,
            shape_vector=False
    ):
        super().__init__()

        self.shape = shape
        self.shape_vector = shape_vector

        self.pos_emb = SinusoidalPosEmb3D(pos_dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, 4 * pos_dim),
            nn.GELU(),
            nn.Linear(4 * pos_dim, out_channels)
        )
    
    def forward(self, frame_indices):

        B, device = frame_indices.shape[0], frame_indices.device
        H, W, T = self.shape

        x_coords = torch.arange(W, device=device).view(1, 1, 1, W).expand(1, 1, H, W)  # [1, H, W, 1]
        y_coords = torch.arange(H, device=device).view(1, 1, H, 1).expand(1, 1, H, W)  # [1, H, W, 1]

        x_coords = x_coords.expand(B, T, H, W)  # [B, T, H, W]
        y_coords = y_coords.expand(B, T, H, W)  # [B, T, H, W]

        frame_indices = repeat(frame_indices, 'b t -> b t h w', h=H, w=W)
        
        positional_embedding = torch.stack((x_coords, y_coords, frame_indices), dim=-1)  # [B, T, H, W, 3]
        
        coordinates = rearrange(positional_embedding, 'b t h w c -> (b h w t) c')
        pos_emb3D = self.pos_emb(coordinates)
        pos_emb3D = self.pos_mlp(pos_emb3D)
        
        pos_emb3D = rearrange(pos_emb3D, '(b h w t) c -> b c t h w', b=B, h=H, w=W, t=T)
        return pos_emb3D
    
def build_2d_sincos_position_embedding(h, w, embed_dim, temperature=10000.):
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed

class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.GateConv = nn.Conv2d(in_channels+hidden_channels, 2*hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.NewStateConv = nn.Conv2d(in_channels+hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, inputs, prev_h):
        """
        inputs: (N, in_channels, H, W)
        Return:
            new_h: (N, hidden_channels, H, W)
        """
        gates = self.GateConv(torch.cat((inputs, prev_h), dim = 1))
        u, r = torch.split(gates, self.hidden_channels, dim = 1)
        u, r = F.sigmoid(u), F.sigmoid(r)
        h_tilde = F.tanh(self.NewStateConv(torch.cat((inputs, r*prev_h), dim = 1)))
        new_h = (1 - u)*prev_h + h_tilde

        return new_h


class MotionEncoder(nn.Module):
    """
    Modified from
    https://github.com/sunxm2357/mcnet_pytorch/blob/master/models/networks.py
    """
    def __init__(self, in_channels, model_channels, n_downs=2):
        super().__init__()
        ch = model_channels
        model = []
        model += [nn.Conv2d(in_channels, ch, 5, padding = 2)]
        model += [nn.ReLU()]
        
        for i in range(n_downs-1):
            # if i == 0:
            model += [nn.MaxPool2d(2)]
            model += [nn.Conv2d(ch, ch*2, 5, padding = 2)]
            ch *= 2
            model += [nn.ReLU()]
        
        model += [nn.MaxPool2d(2)]
        model += [nn.Conv2d(ch, ch * 2, 7, padding = 3)]
        model += [nn.ReLU()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        out: (B, C*(2^n_downs), H//(2^n_downs), W//(2^n_downs))
        """
        out = self.model(x)
        return out

class MotionConditioning(nn.Module):
    def __init__(self, in_channels, model_channels, n_downs):
        super().__init__()
        
        self.model_channels = model_channels
        self.motion_dim = model_channels * 2**n_downs
        self.motion_encoder = MotionEncoder(in_channels, model_channels, n_downs,)
        self.conv_gru_cell = ConvGRUCell(self.motion_dim, self.motion_dim, kernel_size = 3, stride = 1, padding = 1)
        pass
    
    def global_context_encode(self, x):
        B, C, T, H, W = x.shape
        
        x = rearrange(x, 'b c t h w -> b t c h w')
        x = unnormalize_img(x)
        diff_images = normalize_img(x[:, 1:, ...] - x[:, 0:-1, ...]) #(B, T-1, C, H, W)
    
        h = self.condition_enc(diff_images) #(B, T-1, C, H, W)

        m = torch.zeros(h.shape[-3:], device = h.device)
        m = repeat(m, 'C H W -> B C H W', B=B)
        
        #update m given the first observed frame conditional feature
        m = self.conv_gru_cell(h[:, 0, ...], m)

        #recurrently calculate the context motion feature by GRU
        To = h.shape[1]
        for i in range(1, To):
            m = self.conv_gru_cell(h[:, i, ...], m)

        return m

    
    def condition_enc(self, x):
        B, T, _, _, _ = x.shape
        x = x.flatten(0, 1)
        x = self.motion_encoder(x)
        
        return rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T)
    
    
class MotionPredictor(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        attn_dim_head=32,
        depth=8,
        tc=10,
        tp=5,
        num_classes=6,
    ):
        super().__init__()
        self.tc = tc
        self.tp = tp
        self.proj = nn.Conv2d(dim, out_dim, 1)
        
        attn_heads = out_dim // attn_dim_head
        
        self.blocks = nn.ModuleList([])
        
        for _ in range(depth):
            self.blocks.append(nn.ModuleList([
                AttentionLayer(out_dim, heads=attn_heads, dim_head=attn_dim_head),
                CrossAttentionLayer(out_dim, context_dim=out_dim, heads=attn_heads, dim_head=attn_dim_head),
            ]))
        self.frame_mlp = nn.Sequential(
            SinusoidalPosEmb(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.final_conv = zero_init(nn.Conv2d(out_dim, out_dim, 1))
        
        self.class_emb = nn.Embedding(num_classes, out_dim) if num_classes else None
        
    def forward(self, motion_feature, frame_indices, class_cond):
        """_summary_
        Args:
            motion_feature (_type_): [B, C, H', W']
            frame_indices (_type_): [B, T]
            action_emb (_type_): [B, C]
        """
        B, C, H, W = motion_feature.shape
        
        cond_frame_idx, pred_frame_idx = frame_indices[:,:self.tc], frame_indices[:,self.tc:]
        frame_distance = pred_frame_idx[:,  0] - cond_frame_idx[:, 0]
        
        frame_emb = self.frame_mlp(frame_distance).unsqueeze(1) # [B, 1, C]
        action_emb = self.class_emb(class_cond).unsqueeze(1) if self.class_emb else None # [B, 1, C]
        frame_action_emb = torch.cat([frame_emb, action_emb], dim=1)
        
        motion_feature = self.proj(motion_feature)
        
        motion_feature = rearrange(motion_feature, 'b c h w -> b (h w) c')
        
        x = motion_feature
        for self_attn, cross_attn in self.blocks:
            x = self_attn(x)
            x = cross_attn(x, context=frame_action_emb)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        out = self.final_conv(x)
        return out        
        