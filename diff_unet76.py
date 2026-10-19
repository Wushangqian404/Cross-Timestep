import math
import copy
from random import random
from functools import partial
from collections import namedtuple

from beartype import beartype
import torch

from torch import einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.amp import autocast
from einops import rearrange
from einops.layers.torch import Rearrange
import math
import torch.nn as nn
from tqdm.auto import tqdm
import numpy
from monai.losses import DiceCELoss
# from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules

class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv3dBN(nn.Module):
    """带 BN+GELU 的 3D 卷积"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv3d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm3d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv3d(Conv3dBN):
    """3D 深度可分离卷积"""

    def __init__(self, c1, c2, k=3, s=1, d=1, act=False):
        # groups 取 gcd(c1,c2) 以保持 depthwise
        super().__init__(c1, c2, k, s, None, math.gcd(c1, c2), d, act)


class CMRF3D(nn.Module):
    """3D 版轻量级级联多感受野模块"""

    def __init__(self, c1, c2, N=8, shortcut=True, e=0.5):
        super().__init__()
        self.N = N
        self.c = max(1, int(c2 * e / N))
        self.add = shortcut and (c1 == c2)

        # 1×1×1 降维
        self.pw1 = Conv3dBN(c1, 2 * self.c, k=1, act=True)
        # 最后再用 1×1×1 升维
        self.pw2 = Conv3dBN(N * self.c, c2, k=1, act=True)

        # 除第一段外，剩余 N-1 段都用深度卷积
        self.m = nn.ModuleList(DWConv3d(self.c, self.c, k=3, act=False) for _ in range(N - 1))

    def forward(self, x):
        # with autocast(device_type="cuda"):
        identity = x
        # 降维得到 shape=(B, c2//N, D,H,W)
        x = self.pw1(x)

        # 按通道奇偶拆两份
        # chunks = [x[:, 0::2], x[:, 1::2]]
        # # 对第二份继续级联深度卷积
        # for conv in self.m:
        #     chunks.append(conv(chunks[-1]))
        # # 将第一份与第二份做一次加和，再移除第二份
        # chunks[0] = chunks[0] + chunks[1]
        # chunks.pop(1)
        # # 拼回去
        # y = torch.cat(chunks, dim=1)
        x0, x1 = x[:, 0::2], x[:, 1::2]
        y = [x0 + x1]  # 初始块
        current = x1

        for conv in self.m:  # 逐层计算，覆盖中间结果
            current = conv(current)
            y.append(current)

        y = torch.cat(y, dim=1)

        y = self.pw2(y)
        return (identity + y) if self.add else y


# class DepthwiseSeparableConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super().__init__()
#         # 深度卷积（分组卷积）
#         self.depthwise = nn.Conv3d(
#             in_channels,
#             in_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             groups=in_channels  # 关键：分组数=输入通道数
#         )
#         # 逐点卷积（1x1x1卷积）
#         self.pointwise = nn.Conv3d(
#             in_channels,
#             out_channels,
#             kernel_size=1,
#             stride=1,
#             padding=0
#         )
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x


# class Downsample(nn.Module):
#     def __init__(self, in_channels, out_channels=None):
#         super().__init__()
#         out_channels = out_channels or in_channels
#         self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.fuse = CMRF3D(in_channels, out_channels)
#
#     def forward(self, x):
#         x = self.fuse(x)
#         x = self.pool(x)
#         return x
#
#
# class Upsample(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
#         self.fuse = CMRF3D(in_channels, out_channels)
#
#     def forward(self, x, skip_shape):
#         x = self.upsample(x)
#         # 对 x 做裁剪，使其和 skip 空间尺寸一致
#         Ds, Hs, Ws = skip_shape[2], skip_shape[3], skip_shape[4]
#         x = x[..., :Ds, :Hs, :Ws]
#         x = self.fuse(x)
#         return x


# class Upsample(nn.Module):
#     def __init__(self, dim, dim_out=None):
#         super().__init__()
#         dim_out = dim_out or dim
#         self.deconv = DepthwiseSeparableConv3d(dim,dim_out,kernel_size=3,stride=2,padding=1)
#
#         self.norm   = nn.GroupNorm(num_groups=1, num_channels=dim_out)
#         self.act    = nn.SiLU()
#
#     def forward(self, x, skip_shape):
#         x = self.deconv(x)
#         # 根据 skip 的真实空间尺寸做裁剪
#         Ds, Hs, Ws = skip_shape[2], skip_shape[3], skip_shape[4]
#         x = x[..., :Ds, :Hs, :Ws]
#         return self.act(self.norm(x))
class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        # self.conv = DepthwiseSeparableConv3d(dim, dim_out, kernel_size=3, padding=1)  # 普通卷积非转置
        self.conv = CMRF3D(dim, dim_out)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.act = nn.SiLU()

    def forward(self, x, skip_shape):
        # 使用插值实现精确2倍上采样
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
        x = self.conv(x)  # 调整通道数
        _, _, Ds, Hs, Ws = skip_shape
        x = x[..., :Ds, :Hs, :Ws]
        # print(f"ssu:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        return self.act(self.norm(x))





class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        # self.conv = DepthwiseSeparableConv3d(dim,dim_out,kernel_size=3,stride=2,padding=1)
        self.conv = nn.Sequential(
            # DepthwiseSeparableConv3d(dim, dim_out, kernel_size=3, stride=2, padding=1),
            nn.Conv3d(dim, dim_out, kernel_size=3, stride=2, padding=1),
            CMRF3D(dim_out, dim_out)
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.act  = nn.SiLU()

    def forward(self, x):
        # 动态 padding，确保 D/H/W 都为偶数
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        pad_d = (0, D % 2)
        pad_h = (0, H % 2)
        pad_w = (0, W % 2)
        # F.pad 参数顺序: (W_left, W_right, H_left, H_right, D_left, D_right)
        x = F.pad(x, pad_w + pad_h + pad_d)
        x = self.conv(x)
        # print(f"ssd:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        return self.act(self.norm(x))


class lstmNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = 1e-5

    def forward(self, x):
        # print(f"1当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        x = F.layer_norm(
            x,
            normalized_shape=(x.size(-1),),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps
        )
        # print(f"11当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        return x


class cond_lstmNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = 1e-5

    def forward(self, x):
        x = F.layer_norm(
            x,
            normalized_shape=(x.size(-1),),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps
        )
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = 1e-5

    def forward(self, x):
        # 输入形状 [B, C, D, H, W]
        B, C, D, H, W = x.shape
        t = x.view(B, C, -1)  # [B, C, D*H*W]
        mean = x.mean(dim=-1, keepdim=True)  # [B, C, 1]
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(B, C, D, H, W)  # 恢复原形状
        x = x * self.weight.view(1, C, 1, 1, 1)
        if self.bias is not None:
            x += self.bias.view(1, C, 1, 1, 1)
        return x


class SinusoidalPosEmb(Module):
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


# building block modules

class Block(Module):
    # def __init__(self, dim, dim_out, groups=8):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        # self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.proj = nn.Conv3d(dim, dim_out, 3, padding=1)
        # self.proj = DepthwiseSeparableConv3d(dim,dim_out,kernel_size=3,stride=1,padding=1)
        # self.proj = CMRF3D(dim, dim_out)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        with autocast(device_type="cuda"):
            x = self.proj(x)
            x = self.norm(x)

            if exists(scale_shift):
                scale, shift = scale_shift
                x = x * (scale + 1) + shift

            x = self.act(x)
            return x


class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        # def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        # self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.res_conv = nn.Conv3d(dim, dim_out, 3,padding=1) if dim != dim_out else nn.Identity()
        # self.res_conv = DepthwiseSeparableConv3d(dim,dim_out,kernel_size=3,stride=1,padding=1) if dim != dim_out else nn.Identity()

        # self.res_conv = CMRF3D(dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
            scale_shift = None
            if exists(self.mlp) and exists(time_emb):
                time_emb = self.mlp(time_emb)
                # time_emb = rearrange(time_emb, 'b c -> b c 1 1')
                time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
                scale_shift = time_emb.chunk(2, dim=1)

            h = self.block1(x, scale_shift=scale_shift)

            h = self.block2(h)

            return h + self.res_conv(x)


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv3d(dim, inner_dim, 3,padding=1),
        # DepthwiseSeparableConv3d(dim, inner_dim,3,1,1),
        # CMRF3D(dim, inner_dim),
        nn.GELU(),
        # CMRF3D(inner_dim, dim),
        # DepthwiseSeparableConv3d(inner_dim, dim,3,1,1)
        nn.Conv3d(inner_dim, dim, 3,padding=1),
    )


class CBAM3D(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM3D, self).__init__()
        if (reduction_ratio > channels):
            reduction_ratio = channels
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1),
            nn.BatchNorm3d(channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1),
            nn.BatchNorm3d(channels),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # with autocast(device_type="cuda"):   # Channel Attention
        ca = self.channel_attention(x)
        x = x * ca
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x


class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        self.ln = nn.LayerNorm(3 * hidden_size)

    def forward(self, x, h):
        # with autocast(device_type="cuda"):
        # print(f"x:{x.shape},h:{h.shape}")
        # print(f"gru:")
        #
        # print(x[0][0])
        #
        # print(h[0][0])
        combined = torch.cat((x, h), dim=2)
        gates = self.linear(combined)

        # print(gates[0][0])
        gates = self.ln(gates)

        # print(gates[0][0])
        r_gate, z_gate, n_gate = gates.chunk(3, dim=2)
        del combined, gates
        torch.cuda.empty_cache()
        r_gate = torch.sigmoid(r_gate)
        z_gate = torch.sigmoid(z_gate)
        n_gate = torch.tanh(n_gate + r_gate * h)

        h_next = (1 - z_gate) * n_gate + z_gate * h
        return h_next


class ConvLSTM3DCell(nn.Module):
    def __init__(self, input_dim, hidden_size, kernel_size=3, padding=1, last_layer=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.last_layer = last_layer
        # self.conv = nn.Sequential(
        # # 深度卷积（空间特征提取）
        #     nn.Conv3d(input_dim + hidden_size, input_dim + hidden_size, kernel_size=3, padding=1,
        #               groups=input_dim + hidden_size),  # groups=dim实现通道独立
        #     # 逐点卷积（通道特征融合）
        #     nn.Conv3d(input_dim + hidden_size, 4 * hidden_size, kernel_size=1)  # 1x1x1卷积调整通道
        # )
        # self.conv = DepthwiseSeparableConv3d(input_dim + hidden_size, 4 * hidden_size,3,1,1)
        self.conv = nn.Conv3d(input_dim + hidden_size, 4 * hidden_size, 3, 1,1)
        # self.conv = DWConv3d(input_dim + hidden_size, 4 * hidden_size, 3, 1, 1)
        # self.conv = CMRF3D(input_dim + hidden_size, 4 * hidden_size,N=4,e=0.25)
        if (last_layer):
            self.proj = nn.Conv3d(in_channels=input_dim,
                                  out_channels=hidden_size,
                                  kernel_size=1
                                  )
        self.bn = nn.GroupNorm(1, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        # with autocast(device_type="cuda"):
        if (self.last_layer):
            h_prev = self.proj(h_prev)
            c_prev = self.proj(c_prev)
        combined = torch.cat([x, h_prev], dim=1)
        # print(f"3:x:{x.shape},h_t:{h_prev.shape},c_t:{c_prev.shape}")
        # print(f"convlstm:")

        # print(x[0][0][0][0])

        with autocast(device_type="cuda", dtype=torch.float32):
            # print(h_prev[0][0][0][0])
            # print(combined[0][0][0][0])
            conv_output = self.conv(combined)

            # print(conv_output[0][0][0][0])
            conv_output = self.bn(conv_output)
            # print(conv_output[0][0][0])
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        del combined, conv_output
        torch.cuda.empty_cache()
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # print(i[0][0][0][0])
        #
        # print(f[0][0][0][0])
        #
        # print(o[0][0][0][0])
        #
        # print(g[0][0][0][0])
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class HybridConvLSTM3D(nn.Module):
    def __init__(self, input_dim, time_dim, num_layers=4, hidden_size=64, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.time_proj = nn.Linear(time_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.layers.append(ConvLSTM3DCell(input_dim if i == 0 else hidden_size, hidden_size))
            else:
                self.layers.append(CustomGRUCell(input_size=hidden_size, hidden_size=hidden_size))
        # self.residual_convs = nn.ModuleList([nn.Sequential(
        #         nn.Conv3d(input_dim if i == 0 else hidden_size, hidden_size, kernel_size=1),
        #         nn.GroupNorm(1,hidden_size),
        #         nn.ReLU()
        #     ) for i in range(num_layers)
        # ])
        # self.residual_convs.append(nn.Sequential(nn.Conv3d(hidden_size, input_dim, kernel_size=1),
        #                                          nn.GroupNorm(1, input_dim),
        #                                          nn.ReLU()))
        # self.out = ConvLSTM3DCell(hidden_size, input_dim,last_layer=True)
        self.bn = nn.GroupNorm(1, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.lastnorm = nn.GroupNorm(1, input_dim)

    def forward(self, x, time_emb):
        # with autocast(device_type="cuda"):
        b, c, d, h, w = x.size()
        t = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d, h, w)
        t_seq = t.view(b, self.hidden_size, -1).permute(0, 2, 1)
        h_t = torch.zeros_like(t)
        c_t = torch.zeros_like(t)
        h_t = h_t + t
        # residual = x
        # print(f":hy*")
        # print(x[0][0][0][0])

        # print(time_emb)
        # print(f"self.input_dim:{self.input_dim},x:{x.shape}")
        for i, layer in enumerate(self.layers):
            # residual = self.residual_convs[i](x)
            if isinstance(layer, ConvLSTM3DCell):
                # print(f"x_conv1:{x.shape}")
                # print(f"{i}:hy1")
                # print(x[0][0][0][0])
                x, c_t = layer(x, h_t, c_t)
                # print(f"{i}:hy2.1")
                # print(x[0][0][0][0])
                x = self.dropout(x)
                # print(f"{i}:hy2.2")
                # print(x[0][0][0][0])
                # print(residual[0][0][0][0])
                # x = x + residual  # 添加残差连接
                # print(f"{i}:hy2.3")
                # print(x[0][0][0][0])
                x = self.bn(x)
                h_t = x
                # print(f"{i}:hy2")
                # print(x[0][0][0][0])

                # print(f"x_conv2:{x.shape}")
            else:
                # print(f"x_dim:{x.shape},self.hidden_size:{self.hidden_size}")
                x = x.contiguous().view(b, self.hidden_size, -1).permute(0, 2, 1)
                h_t = h_t.contiguous().view(b, self.hidden_size, -1).permute(0, 2, 1)
                # print(f"x_:{x.shape},h_t:{h_t.shape}")
                x = layer(x, h_t)
                x = x.permute(0, 2, 1).view(b, self.hidden_size, d, h, w)
                # print(f"{i}:hy3")
                # print(x[0][0][0][0])
                # x = x + residual
                x = self.bn(x)
                h_t = x
                # print(f"{i}:hy4")
                # print(x[0][0][0][0])

        # residual = self.residual_convs[self.num_layers](x)
        # print(f"2:x:{x.shape},h_t:{h_t.shape},c_t:{c_t.shape},residual:{residual.shape}")
        # x, c_t = self.out(x, h_t+t, c_t)

        # print(f"{i}:hy5")
        # print(x[0][0][0][0])
        # x = self.dropout(x)
        # x = x + residual  # 添加残差连接

        # print(f"{i}:hy6")
        # print(x[0][0][0][0])
        x = self.lastnorm(x)
        return x


class DLSTM3D(nn.Module):
    def __init__(self, in_channels, time_dim, lstm_layers=4, hidden_size=64, kernel_size=3, padding=1, dropout=0.1):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, hidden_size)
        self.in_channels = in_channels
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        # self.hidden_size = 16
        # 空间分支
        self.spatial_cells = nn.ModuleList([
            ConvLSTM3DCell(
                input_dim=in_channels if i == 0 else hidden_size,
                hidden_size=hidden_size,
            ) for i in range(3)
        ])
        self.spatial_gru = CustomGRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.spatial_last = ConvLSTM3DCell(
            input_dim=hidden_size,
            hidden_size=hidden_size,
        )
        # 通道分支
        self.channel = nn.ModuleList([])
        self.channel.append(CustomGRUCell(input_size=in_channels, hidden_size=hidden_size))
        self.channel.append(ConvLSTM3DCell(input_dim=hidden_size, hidden_size=hidden_size))
        self.channel.append(CustomGRUCell(input_size=hidden_size, hidden_size=hidden_size))
        self.channel.append(ConvLSTM3DCell(input_dim=hidden_size, hidden_size=hidden_size))

        # self.residual_convs = nn.ModuleList([
        #     nn.Conv3d(in_channels if i == 0 else hidden_size, hidden_size, kernel_size=1)
        #     for i in range(5)
        # ])

        # 简短注意力，可用CBAM3D或1×1×1卷积门
        self.attn = CBAM3D(channels=2 * hidden_size)

        # 最终投影回原通道数
        self.out_proj = nn.Conv3d(2 * hidden_size, in_channels, kernel_size=1)
        self.bn = nn.GroupNorm(1, hidden_size)
        self.lastnorm = nn.GroupNorm(1, in_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # with autocast(device_type="cuda"):
        B, C, D, H, W = x.shape
        # 时间嵌入扩展
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D, H, W)
        t_seq = t.permute(0, 3, 4, 2, 1).reshape(B, H * W * D, -1)
        spatial_h = torch.zeros_like(t)
        channel_h = torch.zeros_like(t)
        spatial_h = spatial_h + t
        channel_h = channel_h + t
        # 空间分支前3层 ConvLSTM
        s = x
        c_state = torch.zeros_like(spatial_h)
        i = 0

        # print(f"ab1:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        for cell in self.spatial_cells:
            # residual = self.residual_convs[i](s)
            # tmp=spatial_h+t
            # print(f"s:{s.shape},spatial_h+t:{tmp.shape}")
            s, c_state = cell(s, spatial_h, c_state)
            spatial_h = s
            s = self.dropout(s)
            # s = s+residual
            s = self.bn(s)
            i = i + 1

        # print(f"ab2:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        # 空间分支最后GRU
        # residual = self.residual_convs[3](s)
        s = s.view(B, self.hidden_size, -1).permute(0, 2, 1)
        spatial_h = spatial_h.view(B, self.hidden_size, -1).permute(0, 2, 1)
        s = self.spatial_gru(s, spatial_h)
        s = s.permute(0, 2, 1).view(B, self.hidden_size, D, H, W)
        spatial_h = s
        s = self.dropout(s)
        # s = s + residual

        # print(f"ab3:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        s = self.bn(s)
        # residual = self.residual_convs[4](s)
        s, c_state = self.spatial_last(s, spatial_h, c_state)
        # spatial_out = s + residual

        # print(f"ab4:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        spatial_out = s
        del s
        torch.cuda.empty_cache()
        # 通道分支
        ch = x
        for i, layer in enumerate(self.channel):
            # residual = self.residual_convs[i](ch)
            if isinstance(layer, ConvLSTM3DCell):
                # print(f"x_conv1:{x.shape}")
                ch, c_state = layer(ch, channel_h , c_state)
                channel_h = ch
                ch = self.dropout(ch)
                # ch = ch + residual  # 添加残差连接
                ch = self.bn(ch)
                # print(f"x_conv2:{x.shape}")
                # print(f"{i}*ab:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            else:
                # print(f"x_dim:{x.shape},self.hidden_size:{self.hidden_size}")
                ch = ch.contiguous().permute(0, 3, 4, 2, 1)
                ch = ch.reshape(B, H * W * D, -1)
                # if i == 0:
                #     # ch = ch.contiguous().view(B, self.in_channels, -1).permute(0, 2, 1)
                #
                # else:
                #     ch = ch.contiguous().view(B, self.hidden_size, -1).permute(0, 2, 1)
                channel_h = channel_h.contiguous().permute(0, 3, 4, 2, 1).reshape(B, H * W * D, -1)
                # print(f"ch:{ch.shape},h:{channel_h.shape},t:{t_seq.shape}")
                ch = layer(ch, channel_h )
                ch = ch.reshape(B, H, W, D, -1).permute(0, 4, 3, 1, 2)
                # ch = ch + residual
                ch = self.dropout(ch)
                ch = self.bn(ch)
                channel_h = ch
                # print(f"{i}*ab:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        channel_out = ch
        del ch
        torch.cuda.empty_cache()
        # 融合并注意力
        fused = torch.cat([spatial_out, channel_out], dim=1)

        a = self.attn(fused)

        out = self.out_proj(a)  # [B, C, D, H, W]

        out = self.lastnorm(out)
        return out




class Transformer(Module):
    def __init__(
            self,
            dim,
            depth=1,
            lstm_layers=4,
            hidden_size=32,
    ):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                # Residual(Attention(dim, dim_head = dim_head, heads = heads)),
                Residual(CBAM3D(dim)),
                Residual(FeedForward(dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


# vision transformer for dynamic ff-parser
class Conditioning_sample(Module):
    def __init__(
            self,
            fmap_size,
            dim,
            time_dim,
            dynamic=True,
            lstm_layers: int = 8,
            hidden_size: int = 64,
            dim_head=32,
            image_size=None,
            heads=4,
            depth=4,
            patch_size=16
    ):
        super().__init__()
        # 主路径调制参数
        self.main_ff_parser_attn_map = nn.Parameter(torch.ones(dim, fmap_size, fmap_size, fmap_size))
        self.dynamic = dynamic

        if dynamic:
            # 动态调制生成器
            self.to_dynamic_main_attn_map = HybridConvLSTM3D(
                input_dim=hidden_size,
                hidden_size=hidden_size,
                time_dim=time_dim,
                num_layers=lstm_layers
            )

        # 特征归一化层
        self.norm_main = LayerNorm(dim, bias=True)
        self.norm_condition = LayerNorm(dim, bias=True)

        # 动态调制投影
        self.dynamic_proj = nn.Conv3d(hidden_size, dim, kernel_size=1)
        self.gn = nn.GroupNorm(8, dim * 4)
        self.gn2 = nn.GroupNorm(4, dim)

        # 特征压缩器
        self.compressor = nn.Sequential(
            nn.Conv3d(dim * 4, dim * 2, kernel_size=1),
            nn.GroupNorm(4, dim * 2),
            nn.GELU(),
            nn.Conv3d(dim * 2, dim, kernel_size=1),
            nn.GroupNorm(2, dim),
            nn.GELU(),
            nn.Conv3d(dim, hidden_size, kernel_size=1),
            nn.GroupNorm(1, hidden_size),
            nn.GELU()
        )

        # 条件调制后处理
        self.post_modulation = ResnetBlock(dim * 2, dim)  # 输入通道加倍

    def forward(self, x, c, time_emb):
        """
        x: 主路径特征 (带噪图像特征)
        c: 条件特征 (清晰图像特征)
        time_emb: 时间嵌入
        """
        B, C, D, H, W = x.shape
        dtype = x.dtype

        # 1. 条件特征频域转换 (使用清晰的图像特征)
        c_freq = torch.fft.fftn(c, dim=(-3, -2, -1))

        # 2. 主路径特征频域转换 (带噪图像特征)
        x_freq = torch.fft.fftn(x, dim=(-3, -2, -1))

        # 3. 初始化静态注意力图
        attn_map = F.interpolate(
            self.main_ff_parser_attn_map.unsqueeze(0),
            size=(D, H, W),
            mode="trilinear",
            align_corners=False
        ).squeeze(0)

        # 4. 动态调制生成
        if self.dynamic:
            # 动态调制输入：使用条件特征的频域信息
            c_real = torch.view_as_real(c_freq)
            x_real = torch.view_as_real(x_freq)
            c_real = rearrange(c_real, 'b c d h w ri -> b (c ri) d h w')
            x_real = rearrange(x_real, 'b c d h w ri -> b (c ri) d h w')

            # 通过压缩器和动态生成器
            to_dynamic_input = torch.cat((x_real, c_real), dim=1)
            to_dynamic_input = self.gn(to_dynamic_input)
            # print("*dynamic_input*")
            # print()
            dynamic_input = self.compressor(to_dynamic_input)



            dynamic_attn_map = self.to_dynamic_main_attn_map(dynamic_input, time_emb)
            dynamic_attn_map = self.dynamic_proj(dynamic_attn_map)
            dynamic_attn_map = self.gn2(dynamic_attn_map)

            # 融合静态和动态注意力图
            attn_map = attn_map + dynamic_attn_map

        # 5. 应用注意力调制到主路径特征
        modulated_x_freq = x_freq * attn_map

        # 6. 反转换回空间域
        modulated_x = torch.fft.ifftn(modulated_x_freq).real.type(dtype)

        # 7. 特征归一化与交叉调制
        normed_x = self.norm_main(modulated_x)
        normed_c = self.norm_condition(c)

        # 核心调制：使用清晰条件特征调制带噪主路径特征
        modulated = modulated_x + (normed_x * normed_c)

        # 8. 与原始特征融合
        fused = torch.cat([x, modulated], dim=1)  # 保留原始信息

        # 9. 调制后处理
        return self.post_modulation(fused)





# class Conditioning_sample(Module):
#     def __init__(
#             self,
#             fmap_size,
#             dim,
#             time_dim,
#             dynamic=True,
#             lstm_layers: int = 8,
#             hidden_size: int = 64,
#             image_size=None,
#             dim_head=32,
#             heads=4,
#             depth=4,
#             patch_size=16
#     ):
#         super().__init__()
#         self.ff_parser_attn_map = nn.Parameter(torch.ones(dim, fmap_size, fmap_size, fmap_size))
#
#         self.dynamic = dynamic
#
#         if dynamic:
#             self.to_dynamic_ff_parser_attn_map = HybridConvLSTM3D(
#                 input_dim=hidden_size,
#                 hidden_size=hidden_size,
#                 time_dim=time_dim,
#                 num_layers=lstm_layers
#             )
#
#         self.norm_input = LayerNorm(dim, bias=True)
#         self.norm_condition = LayerNorm(dim, bias=True)
#         self.dynamic_proj = nn.Conv3d(hidden_size, dim, kernel_size=1)
#         self.block = ResnetBlock(dim, dim)
#         self.gn = nn.GroupNorm(8, dim * 4)
#         self.gn2 = nn.GroupNorm(4, dim)
#         # self.compressor1 = nn.Sequential(
#         #     nn.Conv3d(dim * 4, dim*2, kernel_size=1),  # 4通道(2实+2虚) -> 16通道
#         #     nn.GroupNorm(4, dim*2),
#         #     nn.GELU(),
#         # )
#         # self.compressor2 = nn.Sequential(nn.Conv3d(dim * 2, dim, kernel_size=1),
#         #     nn.GroupNorm(2, dim),
#         #     nn.GELU(),)
#         # self.compressor3 = nn.Sequential(nn.Conv3d(dim, hidden_size, kernel_size=1),
#         #     nn.GroupNorm(1, hidden_size),
#         #     nn.GELU(),)
#         self.compressor = nn.Sequential(
#             nn.Conv3d(dim * 4, dim * 2, kernel_size=1),  # 4通道(2实+2虚) -> 16通道
#             nn.GroupNorm(4, dim * 2),
#             nn.GELU(),
#             nn.Conv3d(dim * 2, dim, kernel_size=1),
#             nn.GroupNorm(2, dim),
#             nn.GELU(),
#             nn.Conv3d(dim, hidden_size, kernel_size=1),
#             nn.GroupNorm(1, hidden_size),
#             nn.GELU()
#         )
#
#     def forward(self, x, c, time_emb):
#         with autocast(device_type="cuda"):
#             # print(f"1-1当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
#             B, C, D, H, W = x.shape
#
#             dtype = x.dtype
#             # print(f"x:{x.shape}")
#             x = torch.fft.fftn(x, dim=(-3, -2, -1))
#             # print(f"cond_x**:")
#             # print(x[0][0][0][0])
#             attn_map = F.interpolate(
#                 self.ff_parser_attn_map.unsqueeze(0),  # [1,C,f,f,f]
#                 size=(D, H, W),
#                 mode="trilinear",
#                 align_corners=False
#             ).squeeze(0)
#             # print(f"attn_map**:")
#             # print(attn_map)
#
#             if self.dynamic:
#                 # with autocast(device_type="cuda"):
#                 # c_complex = fft2(c)
#                 c_complex = torch.fft.fftn(c, dim=(-3, -2, -1))
#
#                 def truncate_fft_3d(tensor, keep_ratio=0.9):
#                     B, C, D, H, W = tensor.shape
#                     device = tensor.device
#                     dtype = tensor.dtype
#
#                     # 1. 计算能量矩阵 (保留原始数据类型)
#                     energy = torch.abs(tensor) ** 2  # [B, C, D, H, W]
#                     total_energy = energy.sum(dim=(-3, -2, -1), keepdim=True)  # [B, C, 1, 1, 1]
#
#                     # 2. 逐样本/通道计算能量阈值
#                     mask = torch.zeros_like(tensor, dtype=torch.bool)
#                     for b in range(B):
#                         for c in range(C):
#                             # 扁平化能量并排序
#                             flat_energy = energy[b, c].flatten()  # [D*H*W]
#                             sorted_energy, indices = torch.sort(flat_energy, descending=True)
#
#                             # 计算累积能量比
#                             cum_energy = torch.cumsum(sorted_energy, dim=0)
#                             condition = (cum_energy >= keep_ratio * total_energy[b, c]).to(torch.long)  # 或 .int()
#                             threshold_idx = torch.argmax(condition)
#
#                             # 生成掩码 (仅保留前threshold_idx个重要频率)
#                             topk_mask = torch.zeros(D * H * W, dtype=torch.bool, device=device)
#                             topk_mask[indices[:threshold_idx]] = True
#                             mask[b, c] = topk_mask.reshape(D, H, W)
#
#                     # 3. 应用掩码并保留原始数据类型
#                     return tensor * mask
#
#                 x = truncate_fft_3d(x, 0.9)  # 显存降低40%
#                 c_complex = truncate_fft_3d(c_complex, 0.9)
#
#                 x_as_real, c_as_real = map(torch.view_as_real, (x, c_complex))
#                 x_as_real, c_as_real = map(lambda t: rearrange(t, 'b c d h w ri -> b (c ri) d h w'),
#                                            (x_as_real, c_as_real))
#                 # print(f"1-2当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
#                 # print(f"x_as_real c_as_real:")
#                 # print(x_as_real)
#                 # print(c_as_real)
#                 to_dynamic_input = torch.cat((x_as_real, c_as_real), dim=1)
#                 # print(f"to_dynamic_input**:")
#                 to_dynamic_input = self.gn(to_dynamic_input)
#                 # print(to_dynamic_input[0][0][0][0])
#
#                 # to_dynamic_input = self.compressor1(to_dynamic_input)
#                 # print(to_dynamic_input[0][0][0][0])
#                 # to_dynamic_input = self.compressor2(to_dynamic_input)
#                 #
#                 # print(to_dynamic_input[0][0][0][0])
#                 # to_dynamic_input = self.compressor3(to_dynamic_input)
#                 # print(to_dynamic_input[0][0][0][0])
#                 to_dynamic_input = self.compressor(to_dynamic_input)
#                 # print(to_dynamic_input[0][0][0][0])
#                 # print(f"2-1当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
#                 # print(f"to_dynamic_input:{to_dynamic_input.shape}")
#                 # print(to_dynamic_input[0][0][0])
#                 dynamic_ff_parser_attn_map = self.to_dynamic_ff_parser_attn_map(to_dynamic_input, time_emb)
#                 # print(f"2-2当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
#                 # print(f"to_dynamic_attn_map**:")
#                 # print(dynamic_ff_parser_attn_map[0][0][0])
#
#                 dynamic_ff_parser_attn_map = self.dynamic_proj(dynamic_ff_parser_attn_map)
#                 dynamic_ff_parser_attn_map = self.gn2(dynamic_ff_parser_attn_map)
#                 attn_map = attn_map + dynamic_ff_parser_attn_map
#                 print(f"2-3当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
#             # print(f"attn_map2**:")
#             # print(attn_map)
#             x = x * attn_map
#             # print(f"cond_x2**:")
#             # print(x[0][0][0][0])
#             # x = ifft2(x).real
#             x = torch.fft.ifftn(x).real
#             x = x.type(dtype)
#             # print(f"cond_x3**:")
#             # print(x[0][0][0][0])
#             # eq 3 in paper
#             # print(f"x:{x.shape}")
#             normed_x = self.norm_input(x)
#             normed_c = self.norm_condition(c)
#             # print(f"c:{c.shape}")
#             # print(f"normed_x:{normed_x.shape}")
#             # print(f"normed_c:{normed_c.shape}")
#             c = (normed_x * normed_c) * c
#             # print(f"cond_c**:")
#             # print(c[0][0][0][0])
#
#             return self.block(c)

# model

@beartype
class Unet(Module):
    def __init__(
            self,
            dim,
            image_size,
            mask_channels=1,
            input_img_channels=3,
            init_dim=None,
            out_dim=None,
            dim_mults: tuple = (1, 2, 4, 8),
            full_self_attn: tuple = (False, False, False, True),
            lstm_layers=8,
            hidden_size=32,
            attn_dim_head=32,
            attn_heads=4,
            mid_transformer_depth=1,
            self_condition=False,
            resnet_block_groups=8,
            conditioning_klass=Conditioning_sample,
            skip_connect_condition_fmaps=False,
            # whether to concatenate the conditioning fmaps in the latter decoder upsampling portion of unet
            dynamic_ff_parser_attn_map=False,
            # allow for ff-parser to be dynamic based on the input. will exclude condition for now
            fft_hidden_size=4,
            conditioning_kwargs: dict = dict(
                lstm_layers=4,
                hidden_size=32,
                dim_head=32,
                heads=4,
                depth=4,
                patch_size=16
            )
    ):
        super().__init__()
        # cp.configure(
        #     mpu=None,  # 如果使用模型并行，则传入相应的 mpu 对象
        #     deepspeed_config=None,  # 如果有 DeepSpeed 配置文件，可以传入
        #     partition_activations=False,  # 是否在多个 GPU 之间分割激活
        #     contiguous_checkpointing=True,  # 是否使用连续内存优化
        #     num_checkpoints=4,  # 激活检查点的数量
        #     checkpoint_in_cpu=True,  # 是否将检查点存储在 CPU 上
        #     synchronize=False,  # 是否在每次检查点调用时同步
        #     profile=False  # 是否启用性能分析
        # )
        self.image_size = image_size

        # determine dimensions

        self.input_img_channels = input_img_channels
        self.mask_channels = mask_channels
        self.self_condition = self_condition

        output_channels = mask_channels
        mask_channels = mask_channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        # self.init_conv = nn.Conv2d(mask_channels, init_dim, 7, padding = 3)
        self.init_conv = nn.Conv3d(mask_channels, init_dim, 7, padding=3)
        # self.cond_init_conv = nn.Conv2d(input_img_channels, init_dim, 7, padding = 3)
        self.cond_init_conv = nn.Conv3d(input_img_channels, init_dim, 7, padding=3)
        self.conditioning_kwargs: dict = dict(
            lstm_layers=lstm_layers,
            hidden_size=fft_hidden_size,
        )
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # print("dims:")
        # print(dims)
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        attn_kwargs = dict(
            lstm_layers=lstm_layers,
            hidden_size=hidden_size,
            dropout=0.1,
        )

        # conditioner settings

        conditioning_klass = partial(
            Conditioning_sample,
            dynamic=dynamic_ff_parser_attn_map,
            **self.conditioning_kwargs
        )

        # layers

        num_resolutions = len(in_out)
        assert len(full_self_attn) == num_resolutions

        self.conditioners = ModuleList([])

        self.skip_connect_condition_fmaps = skip_connect_condition_fmaps

        # downsampling encoding blocks

        self.downs = ModuleList([])

        curr_fmap_size = image_size

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out, full_self_attn)):
            is_last = ind >= (num_resolutions - 1)
            # attn_klass = MambaAttention if full_attn else MambaAttention3D
            attn_klass = DLSTM3D
            self.conditioners.append(
                conditioning_klass(curr_fmap_size, dim_in, time_dim=time_dim, image_size=curr_fmap_size))

            self.downs.append(ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(attn_klass(dim_in, time_dim, **attn_kwargs)),
                # Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
                Downsample(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding=1)
            ]))

            if not is_last:
                curr_fmap_size //= 2

        # middle blocks

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_transformer = Transformer(mid_dim, depth=mid_transformer_depth, lstm_layers=lstm_layers,
                                           hidden_size=hidden_size, )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # condition encoding path will be the same as the main encoding path

        self.cond_downs = copy.deepcopy(self.downs)
        self.cond_mid_block1 = copy.deepcopy(self.mid_block1)
        self.cond_mid_block2 = copy.deepcopy(self.mid_block2)
        self.cond_ups = ModuleList([])
        for ind, ((dim_in, dim_out), _) in enumerate(zip(reversed(in_out), reversed(full_self_attn))):
            is_last = ind == (len(in_out) - 1)
            attn_klass = CBAM3D
            # 跳跃连接只来自条件编码器自身
            skip_connect_dim = dim_in

            self.cond_ups.append(ModuleList([
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                Residual(attn_klass(dim_out)),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv3d(dim_out, dim_in, 3, padding=1)
            ]))

        # --- 条件分支的最终输出层 (生成分割图) ---
        self.seg_final_res_block = block_klass(dim * 2, dim)
        self.seg_final_conv = nn.Conv3d(dim, mask_channels, 1)



        # upsampling decoding blocks

        self.ups = ModuleList([])

        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(in_out), reversed(full_self_attn))):
            is_last = ind == (len(in_out) - 1)
            # attn_klass = MambaAttention if full_attn else MambaAttention3D
            attn_klass = DLSTM3D
            skip_connect_dim = dim_in * (2 if self.skip_connect_condition_fmaps else 1) + dim_out

            self.ups.append(ModuleList([
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                Residual(attn_klass(dim_out, time_dim, **attn_kwargs)),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv3d(dim_out, dim_in, 3, padding=1)
            ]))

        # projection out to predictions

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        # self.final_conv = nn.Conv2d(dim, output_channels, 1)
        self.final_conv = nn.Conv3d(dim, output_channels, 1)


    def forward(
            self,
            x,
            time,
            cond,
            x_self_cond=None
    ):
        assert not torch.isnan(x).any() and not torch.isinf(x).any()
        assert not torch.isnan(cond).any() and not torch.isinf(cond).any()

        dtype, skip_connect_c = x.dtype, self.skip_connect_condition_fmaps
        # print(f"1x:{x.shape}")
        # print(f"1cond:{cond.shape}")
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        # print("*1x:")
        # print(x)
        x = self.init_conv(x)
        # print("*2x:")
        # print(x)
        r = x.clone()

        c = self.cond_init_conv(cond)
        cr = c.clone()
        # print(f"1x:{x.shape}")
        # print(f"1c:{c.shape}")
        # print("*1c:")
        # print(c)
        t = self.time_mlp(time)
        h = []
        h_cond = []
        sum = 0
        # print("*1t:")
        # print(t)
        # print(f"ss3:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        for (block1, block2, attn, downsample), (
        cond_block1, cond_block2, cond_attn, cond_downsample), conditioner in zip(self.downs, self.cond_downs,
                                                                                  self.conditioners):
            x = block1(x, t)
            c = cond_block1(c, t)
            # print(f"{sum}*1xc:")
            # print(x)
            # print(c[0][0][0][0])
            # print(f"ss6:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            h.append([x, c] if skip_connect_c else [x])
            h_cond.append(c)

            x = block2(x, t)
            c = cond_block2(c, t)
            # print(f"2x:{x.shape}")
            # print(f"2c:{c.shape}")
            # print(f"{sum}*2xc:")
            # print(x)
            # print(c[0][0][0][0])
            # print(f"ss7:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            x = attn(x, t_emb=t)
            c = cond_attn(c, t_emb=t)
            # print(f"3x:{x.shape}")
            # print(f"3c:{c.shape}")
            # condition using modulation of fourier frequencies with attentive map
            # you can test your own conditioners by passing in a different conditioner_klass , if you believe you can best the paper
            # print(f"{sum}*3xc:")
            # print(x)
            # print(c[0][0][0][0])
            # print(f"ss8:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            x = conditioner(x, c, t)
            h_cond.append(c)
            # print(f"{sum}*4c:")
            # print(c[0][0][0][0])
            # print(f"ss9:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            h.append([x, c] if skip_connect_c else [x])
            x = downsample(x)
            c = cond_downsample(c)
            # print(f"{sum}*5xc:")
            # # print(x)
            # print(f"{c.shape}")
            sum = sum + 1
            # print(f"123:{sum},当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        x = self.mid_block1(x, t)
        c = self.cond_mid_block1(c, t)

        x = x + c  # seems like they summed the encoded condition to the encoded input representation

        x = self.mid_transformer(x)
        x = self.mid_block2(x, t)
        c = self.cond_mid_block2(c, t)
        # feat3 =   # 最深 mid 输出
        # feat2 = h[-1][0]  # 倒数第1 skip 特征
        # feat1 = h[-3][0]  # 倒数第3 skip 特征
        c_up_features_for_main = []  # 存储注入到主解码器的特征
        for block1, block2, attn, upsample in self.cond_ups:
            c_skip = h_cond.pop()
            c = torch.cat((c, c_skip), dim=1)
            c = block1(c, t)
            c_skip = h_cond.pop()
            c = torch.cat((c, c_skip), dim=1)
            c = block2(c, t)

            c = attn(c)

            # 将这一层的输出存起来，给主解码器用
            c_up_features_for_main.append(c)

            # 使用你提供的 Upsample，需要传入 skip 的 shape

            if len(h_cond) != 0:
                skip_shape = h_cond[-1].shape
                # print(f"skip_shape:{skip_shape}")
                c = upsample(c, skip_shape)
            else:
                c = upsample(c)


        # 生成最终分割掩码
        c = torch.cat((c, cr), dim=1)  # 别忘了最后的残差连接
        # print(f"f_c:{c.shape}")
        c = self.seg_final_res_block(c)
        final_mask_pred = self.seg_final_conv(c)



        sum = 0
        for (block1, block2, attn, upsample),c_up in zip(self.ups,c_up_features_for_main):
            skip = h.pop()
            # if(skip_connect_c):
            #     print(f"1:skip x:{x.shape},skip:{skip[0].shape}")
            # else:
            #     print(f"1:no_skip x:{x.shape},skip:{skip[0].shape}")
            x = torch.cat((x, *skip), dim=1)
            x = torch.cat((x, c_up), dim=1)
            x = block1(x, t)
            # print(f"321-2:{sum},当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            skip = h.pop()
            # if (skip_connect_c):
            #     print(f"2:skip x:{x.shape},skip:{skip[0].shape}")
            # else:
            #     print(f"2:no_skip x:{x.shape},skip:{skip[0].shape}")
            x = torch.cat((x, *skip), dim=1)
            x = torch.cat((x, c_up), dim=1)
            x = block2(x, t)
            x = attn(x, t_emb=t)
            if len(h) != 0:
                skip_shape = h[-1][-1].shape
                # print(f"skip_shape:{skip_shape}")
                x = upsample(x, skip_shape)
            else:
                x = upsample(x)
            sum = sum + 1
            # print(f"321:{sum},当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        final_noise_pred = self.final_conv(x)
        # print(f"**last**当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        # print(f"**last2**当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        # print(f"final_mask:{final_mask.shape}")
        return final_noise_pred , final_mask_pred


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)





class DPMSolverPP:
    def __init__(self, model, medsegdiff_instance, nfe=15, order=2):
        """
        DPM-Solver++(2M)实现
        基于论文：DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models
        """
        self.model = model
        self.medsegdiff = medsegdiff_instance
        self.nfe = nfe
        self.order = order
        self.device = next(model.parameters()).device

        # 预定义时间步（从1到0）
        self.timesteps = torch.linspace(1.0, 0.0, nfe + 1, device=self.device)

    def _model_fn(self, x, t, c):
        """模型包装器：返回预测的x0和噪声"""
        # 将连续时间转换为离散时间索引
        # t_batch = (t * (self.medsegdiff.num_timesteps - 1)).round().long().clamp(0, self.medsegdiff.num_timesteps - 1)
        if t.dtype != torch.long:
            t = t.round().long().clamp(0, self.medsegdiff.num_timesteps - 1)
        pred_x_start, pred_noise, pred_mask = self.medsegdiff.model_predictions(x, t, c,clip_x_start=True)
        return pred_x_start, pred_noise

    def _get_alpha_sigma(self, t, x_shape):
        """
        获取扩展后的alpha和sigma系数
        :param t: 时间步索引 (整数张量)
        :param x_shape: 目标张量的形状
        :return: (alpha, sigma) 扩展后的系数
        """
        # 确保时间步在有效范围内
        t = t.clamp(0, self.medsegdiff.num_timesteps - 1)

        # 提取系数
        alpha = self.medsegdiff.sqrt_alphas_cumprod[t]
        sigma = self.medsegdiff.sqrt_one_minus_alphas_cumprod[t]

        # 扩展维度以匹配目标形状
        # [batch] -> [batch, 1, 1, 1, 1]
        alpha = alpha.view(-1, 1, 1, 1, 1)
        sigma = sigma.view(-1, 1, 1, 1, 1)
        return alpha, sigma

    def dpm_pp_step(self, x, t, t_next, c):
        """
        DPM-Solver++(2M)单步更新
        """
        print(f"**x:{x.shape}")
        print(f"**t1:{t.shape}")
        print(f"**t_next1:{t_next.shape}")
        print(f"**c:{c.shape}")
        t = (t * (self.medsegdiff.num_timesteps - 1)).round().long()
        t_next = (t_next * (self.medsegdiff.num_timesteps - 1)).round().long()

        # print(f"**t2:{t.shape}")
        # print(f"**t_next2:{t_next.shape}")
        # 模型预测
        x0_pred, noise_pred = self._model_fn(x, t, c)
        # print(f"**x0_pred:{x0_pred.shape}")
        # print(f"**noise_pred:{noise_pred.shape}")
        debug_info = {
            'step': f"t={t[0].item():.4f}→t_next={t_next[0].item():.4f}",
            'x_range': f"{x.min().item():.4f}/{x.max().item():.4f}",
            'x0_pred_range': f"{x0_pred.min().item():.4f}/{x0_pred.max().item():.4f}",
            'noise_pred_range': f"{noise_pred.min().item():.4f}/{noise_pred.max().item():.4f}",
            'noise_pred.std()': noise_pred.std().item(),
            'x0_pred.std()': x0_pred.std().item(),
            'SNR_ratio': (x0_pred.std() / noise_pred.std()).item(),
        }
        # 二阶DPM-Solver++更新
        if self.order == 2:
            # 第一次预测
            print(f"**self.medsegdiff.sqrt_alphas_cumprod[t]:{self.medsegdiff.sqrt_alphas_cumprod[t].shape}")
            print(f"**self.medsegdiff.sqrt_one_minus_alphas_cumprod[t]:{self.medsegdiff.sqrt_one_minus_alphas_cumprod[t].shape}")
            alpha_t, sigma_t = self._get_alpha_sigma(t, x.shape)
            x_t = alpha_t * x0_pred + sigma_t * noise_pred

            # print(f"Shapes - x_t: {x_t.shape}, noise: {noise.shape}")
            # print(f"Shapes - alpha_rcp: {self.medsegdiff.sqrt_alphas_cumprod[t]}, alpha_rcp_m1: {self.medsegdiff.sqrt_one_minus_alphas_cumprod[t]}")
            # 第二次预测
            x0_pred_2, noise_pred_2 = self._model_fn(x_t, t_next, c)
            debug_info.update({
                'x_t_range': f"{x_t.min().item():.4f}/{x_t.max().item():.4f}",
                'x0_pred_2_range': f"{x0_pred_2.min().item():.4f}/{x0_pred_2.max().item():.4f}",
                'noise_pred_2_range': f"{noise_pred_2.min().item():.4f}/{noise_pred_2.max().item():.4f}",
                'noise_pred_2.std()': f"{noise_pred_2.std().item():.4f}",
                'x0_pred_2.std()': f"{x0_pred_2.std().item():.4f}",
            })
            # 二阶更新
            alpha_t_next, sigma_t_next = self._get_alpha_sigma(t_next, x.shape)
            x_next = (
                    alpha_t_next * x0_pred_2 + sigma_t_next * noise_pred_2
            )
        else:
            # 一阶更新（Euler方法）
            alpha_t_next, sigma_t_next = self._get_alpha_sigma(t_next, x.shape)
            x_next = alpha_t_next * x0_pred + sigma_t_next * noise_pred
        debug_info['x_next_range'] = f"{x_next.min().item():.4f}/{x_next.max().item():.4f}"
        print(debug_info)

        # return x_next.clamp(-1, 1)
        return x_next

    def sample(self, x, c, progress=True):
        """
        DPM-Solver++采样入口
        """
        # 时间步序列（从噪声到清晰）
        timesteps = self.timesteps
        iterator = zip(timesteps[:-1], timesteps[1:])

        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(timesteps) - 1, desc='DPM-Solver++采样')

        # 按时间步迭代采样
        for t, t_next in iterator:
            # 确保t和t_next是标量值
            t_scalar = t.item() if t.dim() > 0 else t
            t_next_scalar = t_next.item() if t_next.dim() > 0 else t_next

            # 扩展为与batch size相同的形状
            t_expanded = torch.full((x.shape[0],), t_scalar, device=x.device)
            t_next_expanded = torch.full((x.shape[0],), t_next_scalar, device=x.device)

            x = self.dpm_pp_step(x, t_expanded, t_next_expanded, c)

        return x

class MedSegDiff(Module):
    def __init__(
            self,
            model,
            *,
            timesteps=1000,
            sampling_timesteps=None,
            objective='pred_noise',
            beta_schedule='cosine',
            ddim_sampling_eta=1.
    ):
        super().__init__()

        self.model = model if isinstance(model, Unet) else (model.module if hasattr(model, 'module') else model)

        self.input_img_channels = self.model.input_img_channels
        self.mask_channels = self.model.mask_channels
        self.self_condition = self.model.self_condition
        self.image_size = self.model.image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0',
                             'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        alphas_cumprod = alphas_cumprod.clamp(min=1e-6)  # 防止除零错误
        alphas_cumprod_prev = alphas_cumprod_prev.clamp(min=1e-6)

        # 计算recip相关参数时添加稳定性处理
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(torch.clamp(1. / alphas_cumprod - 1, min=1e-6))
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        # register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        register_buffer('log_alphas_cumprod', torch.log(alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    @property
    def device(self):
        return next(self.parameters()).device

    def predict_start_from_noise(self, x_t, t, noise):
        # return (
        #         extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        #         extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        # )
        # alpha_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        # sigma_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # 打印诊断信息
        # print(f"t: {t.min().item()}-{t.max().item()}")
        # print(f"alpha_t range: {alpha_t.min().item():.4e}/{alpha_t.max().item():.4e}")
        # print(f"sigma_t range: {sigma_t.min().item():.4e}/{sigma_t.max().item():.4e}")

        # 数值稳定计算
        # print(f"x_t range: {x_t.min().item():.4f}/{x_t.max().item():.4f}")
        # print(f"noise range: {noise.min().item():.4f}/{noise.max().item():.4f}")
        # print(f"x_start range: {x_start.min().item():.4f}/{x_start.max().item():.4f}")
        # print(f"x_start std: {x_start.std().item():.4f}")
        # return -noise * sigma_t / alpha_t.clamp(min=1e-6)

        # if t.min() > 800:  # 扩散早期
        #     # 当alpha_t很小时，x0 ≈ -noise * sigma_t / alpha_t
        #     # 但我们使用更稳定的近似
        #     return -noise * sigma_t / alpha_t.clamp(min=1e-6)
        #
        #     # 对于中期时间步使用标准稳定公式
        # elif t.min() > 100:
        #     return (x_t - sigma_t * noise) / alpha_t.clamp(min=1e-6)
        #
        #     # 对于后期时间步（低噪声）使用原始公式
        # else:
        #     return (
        #             extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        #             extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        #     )
        # if t.min() > 100:
        #     return (x_t - sigma_t * noise) / alpha_t.clamp(min=1e-6)
        # else:
        #     return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t \
        #         - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

        # 打印结果统计
        alpha_rcp = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        alpha_rcp_m1 = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        x_start = alpha_rcp * x_t - alpha_rcp_m1 * noise
        return x_start

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, c, x_self_cond=None, clip_x_start=False):
        model_output, base_mask = self.model(x, t, c, x_self_cond)
        # print(f"model_output:{model_output.dtype}")
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            # print(f"pred_noise.std().item():{pred_noise.std().item()}")
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            # print(f"1x_start.std().item():{x_start.std().item()}")
            # print(f"1x_start:{x_start.min().item():.4f}/{x_start.max().item():.4f}")
            x_start = maybe_clip(x_start)
            # print(f"2x_start.std().item():{x_start.std().item()}")
            # print(f"2x_start:{x_start.min().item():.4f}/{x_start.max().item():.4f}")
            batch_size = t.shape[0]
            t_normalized = (t.float() / self.num_timesteps)
            mask_weight = torch.exp(-3.0 * (1 - t_normalized))
            mask_weight = mask_weight * (1 - torch.exp(-10.0 * t_normalized))
            mask_weight = mask_weight.view(batch_size, 1, 1, 1, 1) * 0.5  # 时间步权重
            print(f"mask_weight:{mask_weight}")
            # 对base_mask进行后处理
            # processed_mask = torch.sigmoid(base_mask)  # 确保值域[0,1]
            # processed_mask = F.interpolate(processed_mask, size=x_start.shape[2:], mode='trilinear')
            # processed_mask = F.softmax(base_mask)
            # processed_mask = 2. * processed_mask - 1.
            hard_mask = torch.argmax(base_mask, dim=1)
            processed_mask = F.one_hot(hard_mask, num_classes=self.mask_channels).float()
            processed_mask = 2. * rearrange(processed_mask, 'b d h w c -> b c d h w') - 1.
            # 结构增强融合
            x_start = (1-mask_weight)*x_start + mask_weight * processed_mask
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)




        return pred_noise, x_start, base_mask


    def p_mean_variance(self, x, t, c, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, c, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, c, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, c=c,
                                                                          x_self_cond=x_self_cond,
                                                                          clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, cond, self_cond)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, cond_img, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_img, self_cond,
                                                             clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img



    def _model_predict_wrapper(self, x, t, c):
        """包装模型预测函数供DPM-Solver使用"""
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        preds = self.model_predictions(x, t_batch, c)
        return preds.pred_noise  # DPM-Solver需要噪声预测

    @torch.no_grad()

    def dpm_sample(self, cond_img):
        if self.dpm_solver:
            # 使用DPM-Solver++采样
            shape = (cond_img.shape[0], self.mask_channels, *self.image_size)
            return self.dpm_solver.sample(shape, cond_img, steps=15)
        else:
            # 回退到原采样器
            return super().sample(cond_img)

    def q_sample(self, x_start, t, noise):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        
        return x_noisy, noise

    @torch.no_grad()
    def sample(self, cond_img):
        batch_size, device = cond_img.shape[0], self.device
        cond_img = cond_img.to(self.device)
        _, _, D, H, W = cond_img.shape
        print(f"cond_shape{cond_img.shape}")
        if hasattr(self, 'dpm_solver_pp'):
            # 使用DPM-Solver++采样
            return self.dpm_pp_sample(
                (batch_size, self.mask_channels, D, H, W),
                cond_img
            )
        elif hasattr(self, 'dpm_solver'):
            # 使用DPM-Solver-v3采样
            return self.dpm_v3_sample(
                (batch_size, self.mask_channels, D, H, W),
                cond_img
            )
        else:
            # 回退到原始采样方法
            sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
            return sample_fn((batch_size, self.mask_channels, D, H, W), cond_img)


    @torch.no_grad()
    def enable_dpm_pp(self, nfe=15, order=2):
        """启用DPM-Solver++加速采样"""
        self.dpm_solver_pp = DPMSolverPP(
            model=self.model,
            medsegdiff_instance=self,
            nfe=nfe,
            order=order
        )

    @torch.no_grad()
    def dpm_pp_sample(self, shape, cond_img):
        """
        DPM-Solver++采样实现
        """
        device = self.betas.device
        img = torch.randn(shape, device=device)
        return self.dpm_solver_pp.sample(img, cond_img)



    def p_losses(self, x_start, t, cond, noise=None, ac=0.5):
        b, c, d, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                # predicting x_0

                x_self_cond, _ = self.model_predictions(x, t, cond)
                x_self_cond.detach_()

        # predict and take gradient step
        # target = noise
        target = x_start
        model_out, pd_mask = self.model(x, t, cond, x_self_cond)
        # pred_x0 = self.predict_start_from_noise(x, t, model_out).clamp(-1., 1.)、
        pred_x0 = model_out.clamp(-1., 1.)
        pred_noise = self.predict_noise_from_start(x, t, pred_x0)
        # if self.objective == 'pred_noise':
        #     target = target_noise
        # else:
        #     raise ValueError(f'unknown objective {self.objective}')
        bl = 1.0
        main_loss = F.mse_loss(pred_noise, noise)


        time_weight = torch.exp(2 * t.float() / self.num_timesteps)  # 指数增长
        # time_weight = time_weight.view(b, *([1] * (x.dim() - 1)))  # 扩展到特征图维度
        # print(f"time_weight:{time_weight}")
        if main_loss < 0.001:
            bl = 1000
        elif main_loss < 0.01:
            bl = 100
        elif main_loss < 0.1:
            bl = 10
        elif main_loss < 1.0:
            bl = 1
        print(f"1main_loss:{main_loss}")
        main_loss = (main_loss * time_weight).mean()
        print(f"2main_loss:{main_loss}")
        # ce = F.cross_entropy(pd_mask, x_start.argmax(dim=1), reduction='mean')
        # d_loss = DiceCELoss(softmax=True)

        gt_mask = (x_start + 1) / 2.0
        x0_loss = DiceCELoss(softmax=True)(model_out, gt_mask)
        print(f"1x0_loss:{x0_loss}")
        x0_loss = (x0_loss * time_weight).mean()
        print(f"2x0_loss:{x0_loss}")
        # print(f"pd_mask min/max: {pd_mask.min().item():.4f}, {pd_mask.max().item():.4f}")
        # print(f"gt_mask min/max: {gt_mask.min().item():.4f}, {gt_mask.max().item():.4f}")
        mask_loss = DiceCELoss(softmax=True)(
            pd_mask,
            gt_mask
        )
        # mask_loss = mask_loss+ce
        print(f"1mask_loss:{mask_loss}")
        time_weight_mask = torch.exp(2 * t.float() / self.num_timesteps)
        # print(f"time_weight_mask:{time_weight_mask}")
        mask_loss = (mask_loss * time_weight_mask).mean()
        print(f"2mask_loss:{mask_loss}")
        print(f"ac:{ac},bl:{bl}")
        loss = x0_loss + bl * main_loss +  mask_loss
        print(f"p_loss:{loss.item()}")
        return loss

    def forward(self, img, cond_img,epoch, epochs):
        if img.ndim == 4:
            img = rearrange(img, 'b d h w -> b 1 d h w')

        if cond_img.ndim == 4:
            cond_img = rearrange(cond_img, 'b d h w -> b 1 d h w')

        device = self.device
        img, cond_img = img.to(device), cond_img.to(device)

        b, c, d, h, w, device, img_size, img_channels, mask_channels = *img.shape, img.device, self.image_size, self.input_img_channels, self.mask_channels

        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        assert cond_img.shape[1] == img_channels, f'your input medical must have {img_channels} channels'
        assert img.shape[1] == mask_channels, f'the segmented image must have {mask_channels} channels'

        # def temporal_weighted_sampling(sc, num_timesteps, batch_size, device):
        #     progress = epoch / epochs  # 训练进度 [0,1]
        #     t_vals = torch.arange(sc, num_timesteps, device=device).float()
        #
        #     # 1. 基础线性权重（确保中低时间步占比）
        #     linear_weights = (t_vals - sc + 1) / (num_timesteps - sc)
        #
        #     # 2. Sigmoid偏置项（后期强化高时间步）
        #     # 拐点设在总时间步的70%，斜率5.0控制过渡平滑度
        #     sigmoid_bias = 2.0 * progress / (1 + torch.exp(-5.0 * (t_vals - 0.7 * num_timesteps)))
        #
        #     # 3. 组合权重 = 基础权重 + 动态偏置
        #     weights = linear_weights + sigmoid_bias
        #     weights = torch.clamp(weights, min=1e-6)  # 数值稳定性
        #     weights /= weights.sum()
        #
        #     return torch.multinomial(weights, batch_size, replacement=True) + sc
        #
        # if epoch<20 :
        #     sc = int(epoch / epochs * self.num_timesteps)
        # else:
        #     sc = int(numpy.sqrt(epoch / epochs) * self.num_timesteps)
        # sc = min(sc, self.num_timesteps - 200)
        # times = temporal_weighted_sampling(sc, self.num_timesteps, b, device)
        sc = int(numpy.sqrt(epoch / epochs) * self.num_timesteps)
        sc = min(sc, self.num_timesteps - 1)
        times = torch.randint(sc, self.num_timesteps, (b,), device=device).long()
        
        img = normalize_to_neg_one_to_one(img)
        # return self.p_losses(img, times, cond_img, *args, **kwargs)
        # print(f"ss4:,当前显存: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        ac = 0.5
        progress = epoch / epochs
        if progress < 0.25:  # 前25%训练周期
            ac = 0.2
        elif progress < 0.85:  # 25%-65%训练周期
            ac = 0.35
        else:  # 后35%训练周期
            ac = 0.6
        ac = ac * (0.9 + 0.2 * torch.rand(1).item())
        loss = self.p_losses(img, times, cond_img, ac=ac)
        return loss
