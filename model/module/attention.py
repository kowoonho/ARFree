import math
import torch
from torch import nn, einsum

from einops import rearrange, repeat

from model.util import exists, default
from model.module.normalization import Normalization

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=True,
        drop=0.,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=4, dim_head=32, dropout=0., mlp_ratio=4.):
        super().__init__()
        
        self.norm_q = Normalization(query_dim, norm_type='layer')
        if exists(context_dim):
            self.norm_k = Normalization(context_dim, norm_type='layer')
        self.norm_mlp = Normalization(query_dim, norm_type='layer')
        
        self.attn = CrossAttention(query_dim, context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = Mlp(in_features=query_dim, hidden_features=int(mlp_ratio*query_dim))
        
    def forward(self, x, context=None):
        x = x + self.attn(self.norm_q(x), context=self.norm_k(context) if exists(context) else None)
        x = x + self.mlp(self.norm_mlp(x))
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        qkv_fuse = False,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.qkv_fuse = qkv_fuse
        if qkv_fuse:
            self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        else:
            self.to_q = nn.Linear(dim, hidden_dim, bias = False)
            self.to_k = nn.Linear(dim, hidden_dim, bias = False)
            self.to_v = nn.Linear(dim, hidden_dim, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        query,
        key = None,
        value = None,
        pos_bias = None,
    ):
        B, N, C, device = *query.shape, query.device

        if self.qkv_fuse:
            assert key is None and value is None
            qkv = self.to_qkv(query).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            if key is None:
                key = query
            if value is None:
                value = key
            q = rearrange(self.to_q(query), 'b n (h c) -> b h n c', h = self.heads)
            k = rearrange(self.to_k(key), 'b n (h c) -> b h n c', h = self.heads)
            v = rearrange(self.to_v(value), 'b n (h c) -> b h n c', h = self.heads)

        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity
        sim = einsum('b h n d, b h m d -> b h n m', q, k) * self.scale

        # relative positional bias
        if exists(pos_bias):
            if pos_bias.dim() == 3:
                pos_bias = pos_bias.unsqueeze(0)
            mul = sim.shape[0] // pos_bias.shape[0]
            sim = sim + pos_bias.repeat(mul, 1, 1, 1)

        # numerical stability
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('b h n m, b h m d -> b h n d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class AttentionLayer(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=32,
        rotary_emb=None,
        mlp_ratio=4.,
        is_cross=False
    ):
        super().__init__()
        self.norm_q = Normalization(dim, norm_type='layer')
        if is_cross:
            self.norm_k = Normalization(dim, norm_type='layer')
        self.norm_mlp = Normalization(dim, norm_type='layer')
        
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb)
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio*dim))
    
    def forward(self, query, key=None, value=None, pos_bias=None):
        out = query + self.attn(self.norm_q(query), 
                                  key=self.norm_k(key) if key is not None else None,
                                  value=self.norm_k(value) if value is not None else None,
                                  pos_bias=pos_bias)
        out = out + self.mlp(self.norm_mlp(out))
        return out
        
            
        
        
        
        