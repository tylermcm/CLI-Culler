"""MANIQA network definition, for the ranker worker only (imports torch, timm and einops: never import this from the app process).

Adapted from IIGROUP/MANIQA (models/maniqa.py and models/swin.py), Apache License 2.0, Copyright the MANIQA authors,
https://github.com/IIGROUP/MANIQA. Changes: no download of ViT weights (the KonIQ checkpoint already contains them), no hard-coded
``.cuda()``, the per-sample scoring loop is batched, and unused FLOP-counting helpers were dropped. Parameter names are unchanged so the
published checkpoint loads as is.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Block


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, dim_mlp=1024.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim_mlp), act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in slices:
                for w in slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        windows = window_partition(shifted, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(windows, mask=self.attn_mask).view(-1, self.window_size, self.window_size, C)
        shifted = window_reverse(attn_windows, self.window_size, H, W)
        x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted
        x = shortcut + self.drop_path(x.view(B, H * W, C))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7, dim_mlp=1024, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.input_resolution = input_resolution
        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                      shift_size=0 if (i % 2 == 0) else window_size // 2, dim_mlp=dim_mlp, qkv_bias=qkv_bias,
                      qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.input_resolution[0], w=self.input_resolution[1])
        x = F.relu(self.conv(x))
        return rearrange(x, "b c h w -> b (h w) c")


class SwinTransformer(nn.Module):
    def __init__(self, patches_resolution, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), embed_dim=256, drop=0.1,
                 drop_path_rate=0.1, dropout=0.0, window_size=7, dim_mlp=1024, qkv_bias=True, qk_scale=None,
                 attn_drop_rate=0.0, norm_layer=nn.LayerNorm, scale=0.8):
        super().__init__()
        self.scale = scale
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = nn.Dropout(p=drop)
        self.num_features = embed_dim
        self.num_layers = len(depths)
        self.patches_resolution = (patches_resolution[0], patches_resolution[1])
        self.downsample = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            BasicLayer(dim=self.embed_dim, input_resolution=patches_resolution, depth=self.depths[i],
                       num_heads=self.num_heads[i], window_size=self.window_size, dim_mlp=dim_mlp, qkv_bias=qkv_bias,
                       qk_scale=qk_scale, drop=dropout, attn_drop=attn_drop_rate,
                       drop_path=dpr[sum(self.depths[:i]):sum(self.depths[:i + 1])], norm_layer=norm_layer)
            for i in range(self.num_layers)
        ])

    def forward(self, x):
        x = rearrange(self.dropout(x), "b c h w -> b (h w) c")
        for layer in self.layers:
            residual = x
            x = self.scale * layer(x) + residual
        return rearrange(x, "b (h w) c -> b c h w", h=self.patches_resolution[0], w=self.patches_resolution[1])


class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        residual = x
        B, C, N = x.shape
        attn = self.softmax(self.c_q(x) @ self.c_k(x).transpose(-2, -1) * self.norm_fact)
        x = self.proj_drop((attn @ self.c_v(x)).transpose(1, 2).reshape(B, C, N))
        return x + residual


class _SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)


class MANIQA(nn.Module):
    def __init__(self, embed_dim=768, num_outputs=1, patch_size=8, drop=0.1, depths=(2, 2), window_size=4, dim_mlp=768,
                 num_heads=(4, 4), img_size=224, num_tab=2, scale=0.8):
        super().__init__()
        self.input_size = img_size // patch_size
        self.patches_resolution = (self.input_size, self.input_size)
        self.vit = timm.create_model("vit_base_patch8_224", pretrained=False)
        self.save_output = _SaveOutput()
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                layer.register_forward_hook(self.save_output)
        self.tablock1 = nn.ModuleList([TABlock(self.input_size ** 2) for _ in range(num_tab)])
        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(patches_resolution=self.patches_resolution, depths=list(depths),
                                                num_heads=list(num_heads), embed_dim=embed_dim, window_size=window_size,
                                                dim_mlp=dim_mlp, scale=scale)
        self.tablock2 = nn.ModuleList([TABlock(self.input_size ** 2) for _ in range(num_tab)])
        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(patches_resolution=self.patches_resolution, depths=list(depths),
                                                num_heads=list(num_heads), embed_dim=embed_dim // 2, window_size=window_size,
                                                dim_mlp=dim_mlp, scale=scale)
        self.fc_score = nn.Sequential(nn.Linear(embed_dim // 2, embed_dim // 2), nn.ReLU(), nn.Dropout(drop),
                                      nn.Linear(embed_dim // 2, num_outputs), nn.ReLU())
        self.fc_weight = nn.Sequential(nn.Linear(embed_dim // 2, embed_dim // 2), nn.ReLU(), nn.Dropout(drop),
                                       nn.Linear(embed_dim // 2, num_outputs), nn.Sigmoid())

    def forward(self, x):
        self.save_output.outputs.clear()
        self.vit(x)
        taps = [self.save_output.outputs[i][:, 1:] for i in (6, 7, 8, 9)]
        self.save_output.outputs.clear()
        x = torch.cat(taps, dim=2)
        n = self.input_size
        x = rearrange(x, "b (h w) c -> b c (h w)", h=n, w=n)
        for tab in self.tablock1:
            x = tab(x)
        x = self.conv1(rearrange(x, "b c (h w) -> b c h w", h=n, w=n))
        x = self.swintransformer1(x)
        x = rearrange(x, "b c h w -> b c (h w)", h=n, w=n)
        for tab in self.tablock2:
            x = tab(x)
        x = self.conv2(rearrange(x, "b c (h w) -> b c h w", h=n, w=n))
        x = self.swintransformer2(x)
        x = rearrange(x, "b c h w -> b (h w) c", h=n, w=n)
        f = self.fc_score(x).float()
        w = self.fc_weight(x).float()
        return (f * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2))
