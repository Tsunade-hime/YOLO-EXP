import math
import torch
import torch.nn as nn


__all__ = ["CAB", "SAB", "MSCB", "MSCAM"]


def autopad(k, p=None, d=1):
    """自动计算 padding，使 stride=1 时尽量保持特征图尺寸不变"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def channel_shuffle(x, groups=2):
    """
    Channel Shuffle
    输入:  x -> [B, C, H, W]
    输出:  y -> [B, C, H, W]
    """
    b, c, h, w = x.shape
    if groups <= 1 or c % groups != 0:
        return x

    x = x.view(b, groups, c // groups, h, w)      # [B, g, C/g, H, W]
    x = x.transpose(1, 2).contiguous()            # [B, C/g, g, H, W]
    x = x.view(b, c, h, w)                        # [B, C, H, W]
    return x


def get_act(act):
    if act is True or act == "silu":
        return nn.SiLU(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "relu6":
        return nn.ReLU6(inplace=True)
    if act is False or act is None:
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {act}")


class ConvBNAct(nn.Module):
    """基础卷积块：Conv + BN + Act"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = get_act(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))     # [B, c2, H, W]


class ConvBN(nn.Module):
    """基础卷积块：Conv + BN"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.bn(self.conv(x))               # [B, c2, H, W]


class CAB(nn.Module):
    """
    CAB
    Channel Attention Block
    """
    def __init__(self, c1, c2, reduction=16):
        super().__init__()
        self.proj = ConvBNAct(c1, c2, k=1, s=1, act=False) if c1 != c2 else nn.Identity()

        hidden = max(1, c2 // reduction)

        # 保留原始模块结构
        self.fc1 = nn.Conv2d(c2, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, c2, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def _deterministic_adaptive_max_pool2d(self, x, output_size=1):
        """
        当开启 deterministic 且使用 CUDA 时，
        将 adaptive max pool 放到 CPU 上执行，避免触发 warning。
        """
        if x.is_cuda and torch.are_deterministic_algorithms_enabled():
            x_cpu = x.float().cpu() if x.dtype in (torch.float16, torch.bfloat16) else x.cpu()
            y_cpu = torch.nn.functional.adaptive_max_pool2d(x_cpu, output_size)
            return y_cpu.to(device=x.device, dtype=x.dtype)
        return torch.nn.functional.adaptive_max_pool2d(x, output_size)

    def _deterministic_adaptive_avg_pool2d(self, x, output_size=1):
        """
        当开启 deterministic 且使用 CUDA 时，
        将 adaptive avg pool 放到 CPU 上执行，避免后续同类 warning。
        """
        if x.is_cuda and torch.are_deterministic_algorithms_enabled():
            x_cpu = x.float().cpu() if x.dtype in (torch.float16, torch.bfloat16) else x.cpu()
            y_cpu = torch.nn.functional.adaptive_avg_pool2d(x_cpu, output_size)
            return y_cpu.to(device=x.device, dtype=x.dtype)
        return torch.nn.functional.adaptive_avg_pool2d(x, output_size)

    def forward(self, x):
        x = self.proj(x)                                                   # [B, c2, H, W]

        max_feat = self._deterministic_adaptive_max_pool2d(x, 1)           # [B, c2, 1, 1]
        avg_feat = self._deterministic_adaptive_avg_pool2d(x, 1)           # [B, c2, 1, 1]

        max_out = self.fc2(self.relu(self.fc1(max_feat)))                  # [B, c2, 1, 1]
        avg_out = self.fc2(self.relu(self.fc1(avg_feat)))                  # [B, c2, 1, 1]
        attn = self.sigmoid(max_out + avg_out)                             # [B, c2, 1, 1]

        out = x * attn                                                     # [B, c2, H, W]
        return out


class SAB(nn.Module):
    """
    SAB
    Spatial Attention Block

    论文思路：
        Channel Max + Channel Avg
        -> concat
        -> 7x7 Conv
        -> Sigmoid
        -> 与输入逐元素相乘

    参数:
        c1      : 输入通道数
        c2      : 输出通道数
        sab_k   : 空间注意力卷积核大小，论文默认 7

    输入:
        x -> [B, c1, H, W]

    输出:
        y -> [B, c2, H, W]
    """
    def __init__(self, c1, c2, sab_k=7):
        super().__init__()
        self.proj = ConvBNAct(c1, c2, k=1, s=1, act=False) if c1 != c2 else nn.Identity()
        self.conv = nn.Conv2d(2, 1, kernel_size=sab_k, stride=1, padding=sab_k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.proj(x)                           # [B, c2, H, W]

        x_max = torch.max(x, dim=1, keepdim=True)[0]   # [B, 1, H, W]
        x_avg = torch.mean(x, dim=1, keepdim=True)     # [B, 1, H, W]
        attn = torch.cat([x_max, x_avg], dim=1)        # [B, 2, H, W]
        attn = self.sigmoid(self.conv(attn))           # [B, 1, H, W]

        out = x * attn                              # [B, c2, H, W]
        return out


class DWConvBNAct(nn.Module):
    """
    深度卷积块：DWConv + BN + ReLU6
    """
    def __init__(self, c, k=3):
        super().__init__()
        self.conv = nn.Conv2d(
            c, c, kernel_size=k, stride=1, padding=autopad(k),
            groups=c, bias=False
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))     # [B, c, H, W]


class MSDC(nn.Module):
    """
    MSDC
    Multi-scale Depth-wise Convolution

    参数:
        c            : 输入/输出通道数
        kernel_sizes : 多尺度卷积核，论文默认 [1, 3, 5]
        mode         : "parallel" 或 "sequential"
                       论文实验默认使用 parallel

    输入:
        x -> [B, c, H, W]

    输出:
        y -> [B, c, H, W]
    """
    def __init__(self, c, kernel_sizes=(1, 3, 5), mode="parallel"):
        super().__init__()
        assert mode in ("parallel", "sequential")
        self.mode = mode
        self.blocks = nn.ModuleList([DWConvBNAct(c, k=ks) for ks in kernel_sizes])

    def forward(self, x):
        if self.mode == "parallel":
            outs = [blk(x) for blk in self.blocks]             # 每个: [B, c, H, W]
            out = sum(outs)                                    # [B, c, H, W]
        else:
            out = x
            for blk in self.blocks:
                out = out + blk(out)                           # [B, c, H, W]
        return out


class MSCB(nn.Module):
    """
    MSCB
    Multi-scale Convolution Block

    论文思路：
        PWC1 -> BN/ReLU6 -> MSDC -> Channel Shuffle -> PWC2 -> BN
        工程适配版中保留可选 shortcut

    参数:
        c1           : 输入通道数
        c2           : 输出通道数
        expansion    : 扩张比例，论文默认 2
        kernel_sizes : 多尺度深度卷积核，推荐 [1, 3, 5]
        msdc_mode    : "parallel" 或 "sequential"
        shuffle_g    : channel shuffle 分组数
        shortcut     : 是否使用残差

    输入:
        x -> [B, c1, H, W]

    输出:
        y -> [B, c2, H, W]
    """
    def __init__(
        self,
        c1,
        c2,
        expansion=2.0,
        kernel_sizes=(1, 3, 5),
        msdc_mode="parallel",
        shuffle_g=2,
        shortcut=True,
    ):
        super().__init__()
        hidden = max(1, int(c1 * expansion))

        self.pw1 = ConvBNAct(c1, hidden, k=1, s=1, act="relu6")
        self.msdc = MSDC(hidden, kernel_sizes=kernel_sizes, mode=msdc_mode)
        self.shuffle_g = shuffle_g
        self.pw2 = ConvBN(hidden, c2, k=1, s=1)

        self.use_shortcut = shortcut and (c1 == c2)

    def forward(self, x):
        identity = x                                    # [B, c1, H, W]

        x = self.pw1(x)                                 # [B, hidden, H, W]
        x = self.msdc(x)                                # [B, hidden, H, W]
        x = channel_shuffle(x, groups=self.shuffle_g)   # [B, hidden, H, W]
        x = self.pw2(x)                                 # [B, c2, H, W]

        if self.use_shortcut:
            x = x + identity                            # [B, c2, H, W]

        return x


class MSCAM(nn.Module):
    """
    MSCAM
    Multi-Scale Convolutional Attention Module

    论文公式：
        MSCAM(x) = MSCB(SAB(CAB(x)))

    参数:
        c1           : 输入通道数
        c2           : 输出通道数
        reduction    : CAB 压缩率
        sab_k        : SAB 卷积核大小，默认 7
        expansion    : MSCB 扩张比例，默认 2
        kernel_sizes : MSDC 多尺度卷积核，推荐 [1, 3, 5]
        msdc_mode    : "parallel" 或 "sequential"
        shuffle_g    : channel shuffle 分组数
        shortcut     : MSCB 是否使用残差

    输入:
        x -> [B, c1, H, W]

    输出:
        y -> [B, c2, H, W]
    """
    def __init__(
        self,
        c1,
        c2,
        reduction=16,
        sab_k=7,
        expansion=2.0,
        kernel_sizes=(1, 3, 5),
        msdc_mode="parallel",
        shuffle_g=2,
        shortcut=True,
    ):
        super().__init__()

        self.cab = CAB(c1, c2, reduction=reduction)
        self.sab = SAB(c2, c2, sab_k=sab_k)
        self.mscb = MSCB(
            c2, c2,
            expansion=expansion,
            kernel_sizes=kernel_sizes,
            msdc_mode=msdc_mode,
            shuffle_g=shuffle_g,
            shortcut=shortcut,
        )

    def forward(self, x):
        x = self.cab(x)                                 # [B, c2, H, W]
        x = self.sab(x)                                 # [B, c2, H, W]
        x = self.mscb(x)                                # [B, c2, H, W]
        return x


if __name__ == "__main__":
    x = torch.randn(1, 256, 40, 40)

    cab = CAB(c1=256, c2=256, reduction=16)
    y1 = cab(x)
    print("CAB input shape   :", x.shape)               # [1, 256, 40, 40]
    print("CAB output shape  :", y1.shape)              # [1, 256, 40, 40]

    sab = SAB(c1=256, c2=256, sab_k=7)
    y2 = sab(x)
    print("SAB output shape  :", y2.shape)              # [1, 256, 40, 40]

    mscb = MSCB(c1=256, c2=256, expansion=2.0, kernel_sizes=(1, 3, 5), msdc_mode="parallel", shuffle_g=2, shortcut=True)
    y3 = mscb(x)
    print("MSCB output shape :", y3.shape)              # [1, 256, 40, 40]

    mscam = MSCAM(c1=256, c2=256, reduction=16, sab_k=7, expansion=2.0, kernel_sizes=(1, 3, 5), msdc_mode="parallel", shuffle_g=2, shortcut=True)
    y4 = mscam(x)
    print("MSCAM output shape:", y4.shape)              # [1, 256, 40, 40])