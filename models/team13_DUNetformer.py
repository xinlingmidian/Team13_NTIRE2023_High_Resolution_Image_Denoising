import os
import numpy as np
from torch.utils.data import Dataset
import torch
import math
from math import log, pi, sqrt
from functools import partial
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

#####################################Model###################################
List = nn.ModuleList

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth  
# positional embeddings
def apply_rotary_emb(q, k, pos_emb):
    sin, cos = pos_emb
    dim_rotary = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))
    return q, k

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        self.register_buffer('scales', scales)
    def forward(self, x):
        device, dtype, h, w = x.device, x.dtype, *x.shape[-2:]
        seq_x = torch.linspace(-1., 1., steps = h, device = device)
        seq_x = seq_x.unsqueeze(-1)
        seq_y = torch.linspace(-1., 1., steps = w, device = device)
        seq_y = seq_y.unsqueeze(-1)
        scales = self.scales[(*((None,) * (len(seq_x.shape) - 1)), Ellipsis)]
        scales = scales.to(x)
        scales = self.scales[(*((None,) * (len(seq_y.shape) - 1)), Ellipsis)]
        scales = scales.to(x)
        seq_x = seq_x * scales * pi
        seq_y = seq_y * scales * pi
        x_sinu = repeat(seq_x, 'i d -> i j d', j = w)
        y_sinu = repeat(seq_y, 'j d -> i j d', i = h)
        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)
        sin, cos = map(lambda t: rearrange(t, 'i j d -> i j d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'i j d -> () i j (d r)', r = 2), (sin, cos))
        return sin, cos

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))
    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim) 
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, window_size = 16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.window_size = window_size
        inner_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = True)
        self.to_kv = nn.Conv2d(dim, inner_dim*2, 1, bias = True)
        self.to_out = nn.Conv2d(inner_dim, dim, 1,bias = True)
    def forward(self, x, skip = None, time_emb = None, pos_emb = None):
        h, w, b = self.heads, self.window_size, x.shape[0]
        if exists(time_emb):
            time_emb = rearrange(time_emb, 'b c -> b c () ()')
            x = x + time_emb
        q = self.to_q(x)
        kv_input = x
        if exists(skip):
            kv_input = torch.cat((kv_input, skip), dim = 0)
        k, v = self.to_kv(kv_input).chunk(2, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) x y c', h = h), (q, k, v))
        if exists(pos_emb):
            q, k = apply_rotary_emb(q, k, pos_emb)
        q, k, v = map(lambda t: rearrange(t, 'b (x w1) (y w2) c -> (b x y) (w1 w2) c', w1 = w, w2 = w), (q, k, v)) #divisibility
        if exists(skip):
            k, v = map(lambda t: rearrange(t, '(r b) n d -> b (r n) d', r = 2), (k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h x y) (w1 w2) c -> b (h c) (x w1) (y w2)', b = b, h = h, y = x.shape[-1] // w, w1 = w, w2 = w)
        return self.to_out(out)

# class eca_layer(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, channel, k_size=3):
#         super(eca_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
#         self.sigmoid = nn.Sigmoid()
#         self.channel = channel
#         self.k_size =k_size

#     def forward(self, x):
#         # b hw c
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x.transpose(-1, -2))

#         # Two different branches of ECA module
#         y = self.conv(y.transpose(-1, -2))

#         # Multi-scale information fusion
#         y = self.sigmoid(y)

#         return x * y.expand_as(x)
        
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        hidden_dim = dim * mult
        self.project_in = nn.Conv2d(dim, hidden_dim, 1)
        self.project_out = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
            #nn.GELU(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, dim, 1)
        )
        # self.eca = eca_layer(dim)
    def forward(self, x, time_emb = None):
        x = self.project_in(x)
        if exists(time_emb):
            time_emb = rearrange(time_emb, 'b c -> b c () ()')
            x = x + time_emb
        x = self.project_out(x)  
        return x

class Block(nn.Module):
    def __init__(self, dim, depth, dim_head = 64, heads = 8, ff_mult = 4, window_size = 16, time_emb_dim = None, rotary_emb = True):
        super().__init__()
        self.attn_time_emb = None
        self.ff_time_emb = None
        self.pos_emb = AxialRotaryEmbedding(dim_head) if rotary_emb else None
        self.layers = List([])
        for _ in range(depth):
            self.layers.append(List([
                PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads, window_size = window_size)),
                PreNorm(dim, FeedForward(dim, mult = ff_mult))
            ]))
    def forward(self, x, skip = None, time = None):
        attn_time_emb = None
        ff_time_emb = None
        pos_emb = None
        if exists(self.pos_emb):
            pos_emb = self.pos_emb(x)
        for attn, ff in self.layers:
            x = attn(x, skip = skip, time_emb = attn_time_emb, pos_emb = pos_emb) + x
            x = ff(x, time_emb = ff_time_emb) + x
        return x

def padding(input):
    x = input - 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x += 1
    return x - input

# classes
class DUNetformer(nn.Module):
    def __init__(self,dim=16,channels=3,stages=3,num_blocks=1,dim_head=2,window_size=8,heads=2,ff_mult=2,time_emb=False,input_channels=3,output_channels = 3):
        super().__init__()
        input_channels = default(input_channels, channels) 
        output_channels = default(output_channels, channels)
        time_emb_dim = None
        dim_b = dim
        stages_b = 2
        num_blocks_b = 1
        self.fusion = nn.Sequential(nn.Conv2d(dim*2,dim,3,padding=1),
            nn.Sigmoid())
        self.project_in = nn.Sequential(
            nn.Conv2d(input_channels, dim, 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),
        )
        self.project_out = nn.Sequential(
            Block(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim),
            nn.ConvTranspose2d(dim, output_channels, 2, stride = 2)
        )
        self.downs = List([])
        self.ups = List([])
        heads, window_size, dim_head, num_blocks = map(partial(cast_tuple, depth = stages), (heads, window_size, dim_head, num_blocks))
        for ind, heads, window_size, dim_head, num_blocks in zip(range(stages), heads, window_size, dim_head, num_blocks):
            is_last = ind == (stages - 1)
            self.downs.append(List([
                Block(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim),
                nn.Conv2d(dim, dim*2, 4, stride = 2, padding = 1) #1/2
                #nn.PixelUnshuffle(2)
                #nn.Conv2d(dim, dim * 2, 4, stride = 8, padding = 1,dilation=3), #1/8
            ]))
            self.ups.append(List([
                #nn.ConvTranspose2d(dim * 2, dim, 4, stride = 8,padding = 1,dilation=3),
                nn.ConvTranspose2d(dim*2, dim, 2, stride = 2), #2x
                #nn.PixelShuffle(2),
                Block(dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim)
            ]))
            dim *= 2
            if is_last:
                self.mid = nn.Sequential(
                    Block(dim = dim, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim)
                    )  
        ######################breadth####################################
        self.downs_b = List([])
        self.ups_b = List([])
        heads, window_size, dim_head, num_blocks = map(partial(cast_tuple, depth = stages_b), (heads, window_size, dim_head, num_blocks_b))
        for ind, heads, window_size, dim_head, num_blocks in zip(range(stages_b), heads, window_size, dim_head, num_blocks):
            is_last = ind == (stages_b - 1)
            self.downs_b.append(List([
                Block(dim_b, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim),
                nn.Conv2d(dim_b, dim_b * 3, 4, stride = 2, padding = 1) #1/2
                #nn.PixelUnshuffle(2)
                #nn.Conv2d(dim, dim * 2, 4, stride = 8, padding = 1,dilation=3), #1/8
            ]))
            self.ups_b.append(List([
                #nn.ConvTranspose2d(dim * 2, dim, 4, stride = 8,padding = 1,dilation=3),
                nn.ConvTranspose2d(dim_b * 3, dim_b, 2, stride = 2), #2x   nn.ConvTranspose2d(dim_b * 3, dim_b, 2, stride = 2)
                #nn.PixelShuffle(2),
                Block(dim_b, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim),  
            ]))
            dim_b *= 3
            if is_last:
                self.mid_b = nn.Sequential(
                    Block(dim = dim_b, depth = num_blocks, dim_head = dim_head, heads = heads, ff_mult = ff_mult, window_size = window_size, time_emb_dim = time_emb_dim),
                    )
    def forward(self, input, time = None):
        h = input.size(2)
        w = input.size(3)
        h_re = padding(h)
        w_re = padding(w)
        input_data = torch.zeros(1,3,h+h_re,w+w_re).cuda()
        input_data[...,:h,:w] = input
        x = self.project_in(input_data)
        y = x
        skips = []
        for block, downsample in self.downs:
            x = block(x, time = time)
            skips.append(x)
            x = downsample(x) #conv   
        x = self.mid(x)
        for (upsample, block), skip in zip(reversed(self.ups), reversed(skips)):
            x = upsample(x)
            x = block(x, skip = skip)
        ###################  Y   ########################    
        skips_y = []
        for block, downsample in self.downs_b:
            y = block(y, time = time)
            skips_y.append(y)
            y = downsample(y) #conv  
        y = self.mid_b(y) 
        for (upsample, block), skip in zip(reversed(self.ups_b), reversed(skips_y)):
            y = upsample(y)
            y = block(y, skip = skip)
        f = self.fusion(torch.cat([x,y],dim=1))
        x = (f*x) + ((1-f)*y)  
        x = self.project_out(x)    
      
        return x[:,:,:h,:w]

    
