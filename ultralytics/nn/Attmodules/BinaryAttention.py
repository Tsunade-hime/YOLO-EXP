import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import binarize, symquantize

def round_ste(z):
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

class BinaryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attn_quant=False, attn_bias=False, pv_quant=False, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_quant = attn_quant
        self.attn_bias = attn_bias
        self.pv_quant = pv_quant

        if self.attn_bias: # dense bias
            self.input_size = input_size
            self.num_relative_distance = (2 * input_size[0] - 1) * (2 * input_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(input_size[0])
            coords_w = torch.arange(input_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += input_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += input_size[1] - 1
            relative_coords[:, :, 0] *= 2 * input_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(input_size[0] * input_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)

            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    @staticmethod
    def _quantize(x):
        s = x.abs().mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        sign = binarize(x)
        return s * sign
    
    @staticmethod
    def _quantize_p(x):
        qmax = 255
        s = 1.0 / qmax 
        q = round_ste(x / s).clamp(0, qmax)
        return s * q
    
    @staticmethod
    def _quantize_v(x, bits=8):
        act_clip_val = torch.tensor([-2.0, 2.0])
        return symquantize(x, act_clip_val, bits, False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if self.attn_quant:

            q = self._quantize(q)
            k = self._quantize(k)

            attn = (q @ k.transpose(-2, -1)) * self.scale

            if self.attn_bias:
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                            self.input_size[0] * self.input_size[1] + 1,
                            self.input_size[0] * self.input_size[1] + 1, -1)
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
                attn = attn + relative_position_bias.unsqueeze(0)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            if self.pv_quant:
                attn = self._quantize_p(attn)
                v = self._quantize_v(v, 8)

        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x