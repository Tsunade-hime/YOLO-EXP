import math
import torch
import torch.nn as nn


__all__ = ["SRU", "CRU", "SCConv"]


def autopad(k, p=None, d=1):
    """自动计算 padding，使 stride=1 时尽量保持特征图尺寸不变"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible_groups(channels, groups):
    """找到能整除 channels 的合适 group 数"""
    groups = min(groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class ConvBNAct(nn.Module):
    """基础卷积块：Conv + BN + SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))   # [B, c2, H, W]


class SRU(nn.Module):
    """
    SRU
    Spatial Reconstruction Unit

    论文思路：
        1) 通过 GN 的 gamma 生成通道重要性
        2) 经过 sigmoid + threshold 得到 informative / non-informative mask
        3) 将输入分为两部分并进行交叉重构

    参数:
        c1             : 输入通道数
        c2             : 输出通道数
        group_num      : GroupNorm 分组数
        gate_threshold : 门控阈值，论文默认 0.5

    输入:
        x -> [B, c1, H, W]

    输出:
        y -> [B, c2, H, W]
    """
    def __init__(self, c1, c2, group_num=4, gate_threshold=0.5):
        super().__init__()
        self.proj = ConvBNAct(c1, c2, k=1, s=1, act=False) if c1 != c2 else nn.Identity()

        gn_groups = make_divisible_groups(c2, group_num)
        self.gn = nn.GroupNorm(num_groups=gn_groups, num_channels=c2, affine=True)
        self.sigmoid = nn.Sigmoid()
        self.gate_threshold = gate_threshold

        # 交叉重构时需要均分通道
        assert c2 % 2 == 0, "SRU 要求输出通道 c2 为偶数，便于交叉重构"

    def forward(self, x):
        x = self.proj(x)                                        # [B, c2, H, W]

        # -----------------------------
        # 1) GN 标准化，并利用 gamma 生成通道权重
        # -----------------------------
        gn_x = self.gn(x)                                       # [B, c2, H, W]
        gamma = self.gn.weight                                  # [c2]
        w_gamma = gamma / (gamma.sum() + 1e-6)                  # [c2]
        w_gamma = w_gamma.view(1, -1, 1, 1)                     # [1, c2, 1, 1]

        # -----------------------------
        # 2) 生成 informative / non-informative mask
        # -----------------------------
        reweights = self.sigmoid(gn_x * w_gamma)
        info_mask = (reweights >= self.gate_threshold).type_as(x)
        noninfo_mask = (reweights < self.gate_threshold).type_as(x)

        x_info = x * info_mask
        x_noninfo = x * noninfo_mask

        # -----------------------------
        # 3) Cross Reconstruct
        # -----------------------------
        x11, x12 = torch.chunk(x_info, 2, dim=1)                # 每个: [B, c2/2, H, W]
        x21, x22 = torch.chunk(x_noninfo, 2, dim=1)             # 每个: [B, c2/2, H, W]

        y1 = x11 + x22                                          # [B, c2/2, H, W]
        y2 = x12 + x21                                          # [B, c2/2, H, W]

        out = torch.cat([y1, y2], dim=1)                        # [B, c2, H, W]
        return out


class CRU(nn.Module):
    """
    CRU
    Channel Reconstruction Unit

    论文思路：
        Split -> Transform -> Fuse

    参数:
        c1            : 输入通道数
        c2            : 输出通道数
        alpha         : split ratio，论文推荐 1/2
        squeeze_ratio : squeeze ratio，论文典型设置 r=2
        group_size    : GWC 的 group size，论文典型设置 g=2
        group_kernel  : GWC 卷积核大小，默认 3
        act           : 是否使用激活

    输入:
        x -> [B, c1, H, W]

    输出:
        y -> [B, c2, H, W]
    """
    def __init__(
        self,
        c1,
        c2,
        alpha=0.5,
        squeeze_ratio=2,
        group_size=2,
        group_kernel=3,
        act=True,
    ):
        super().__init__()
        assert 0 < alpha < 1, "alpha 必须在 (0, 1) 之间"

        self.proj = ConvBNAct(c1, c2, k=1, s=1, act=False) if c1 != c2 else nn.Identity()

        up_c = max(1, int(round(alpha * c2)))
        low_c = c2 - up_c
        assert low_c > 0, "CRU 中 lower 分支通道必须大于 0"

        up_s = max(1, up_c // squeeze_ratio)
        low_s = max(1, low_c // squeeze_ratio)

        self.up_c = up_c
        self.low_c = low_c

        # -----------------------------
        # 1) Split + Squeeze
        # -----------------------------
        self.squeeze_up = ConvBNAct(up_c, up_s, k=1, s=1, act=act)
        self.squeeze_low = ConvBNAct(low_c, low_s, k=1, s=1, act=act)

        # -----------------------------
        # 2) Upper Transform
        #    GWC + PWC
        # -----------------------------
        gwc_groups = math.gcd(up_s, c2)
        gwc_groups = math.gcd(gwc_groups, group_size)
        gwc_groups = max(gwc_groups, 1)

        self.gwc = ConvBNAct(up_s, c2, k=group_kernel, s=1, g=gwc_groups, act=act)
        self.pwc_up = ConvBNAct(up_s, c2, k=1, s=1, act=act)

        # -----------------------------
        # 3) Lower Transform
        #    PWC + feature reuse
        # -----------------------------
        self.pwc_low = ConvBNAct(low_s, c2 - low_s, k=1, s=1, act=act)

        # -----------------------------
        # 4) Fuse
        #    Pooling + channel-wise softmax
        # -----------------------------
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.proj(x)                                       # [B, c2, H, W]

        # -----------------------------
        # 1) Split
        # -----------------------------
        x_up, x_low = torch.split(x, [self.up_c, self.low_c], dim=1)
        # x_up : [B, up_c, H, W]
        # x_low: [B, low_c, H, W]

        x_up = self.squeeze_up(x_up)                           # [B, up_s, H, W]
        x_low = self.squeeze_low(x_low)                        # [B, low_s, H, W]

        # -----------------------------
        # 2) Upper Transform
        # -----------------------------
        y1 = self.gwc(x_up) + self.pwc_up(x_up)                # [B, c2, H, W]

        # -----------------------------
        # 3) Lower Transform
        # -----------------------------
        y2 = torch.cat([self.pwc_low(x_low), x_low], dim=1)    # [B, c2, H, W]

        # -----------------------------
        # 4) Fuse
        # -----------------------------
        s1 = self.pool(y1)                                     # [B, c2, 1, 1]
        s2 = self.pool(y2)                                     # [B, c2, 1, 1]

        beta = torch.stack([s1, s2], dim=1)                    # [B, 2, c2, 1, 1]
        beta = torch.softmax(beta, dim=1)                      # [B, 2, c2, 1, 1]

        beta1 = beta[:, 0]                                     # [B, c2, 1, 1]
        beta2 = beta[:, 1]                                     # [B, c2, 1, 1]

        out = beta1 * y1 + beta2 * y2                          # [B, c2, H, W]
        return out


class SCConv(nn.Module):
    """
    SCConv
    Spatial and Channel Reconstruction Convolution

    论文公式对应：
        SCConv = SRU -> CRU

    参数:
        c1             : 输入通道数
        c2             : 输出通道数
        alpha          : CRU 的 split ratio
        squeeze_ratio  : CRU 的 squeeze ratio
        group_size     : CRU 中 GWC 的 group size
        group_num      : SRU 中 GroupNorm 的 group 数
        gate_threshold : SRU 中门控阈值
        group_kernel   : CRU 中 GWC 卷积核大小
        act            : 是否使用激活

    输入:
        x -> [B, c1, H, W]

    输出:
        y -> [B, c2, H, W]
    """
    def __init__(
        self,
        c1,
        c2,
        alpha=0.5,
        squeeze_ratio=2,
        group_size=2,
        group_num=4,
        gate_threshold=0.5,
        group_kernel=3,
        act=True,
    ):
        super().__init__()

        self.sru = SRU(c1, c2, group_num=group_num, gate_threshold=gate_threshold)
        self.cru = CRU(
            c2, c2,
            alpha=alpha,
            squeeze_ratio=squeeze_ratio,
            group_size=group_size,
            group_kernel=group_kernel,
            act=act,
        )

    def forward(self, x):
        x = self.sru(x)                                        # [B, c2, H, W]
        x = self.cru(x)                                        # [B, c2, H, W]
        return x


if __name__ == "__main__":
    x = torch.randn(1, 256, 40, 40)

    sru = SRU(c1=256, c2=256, group_num=4, gate_threshold=0.5)
    y1 = sru(x)
    print("SRU input shape   :", x.shape)                      # [1, 256, 40, 40]
    print("SRU output shape  :", y1.shape)                     # [1, 256, 40, 40]

    cru = CRU(c1=256, c2=256, alpha=0.5, squeeze_ratio=2, group_size=2, group_kernel=3, act=True)
    y2 = cru(x)
    print("CRU output shape  :", y2.shape)                     # [1, 256, 40, 40]

    scconv = SCConv(
        c1=256, c2=256,
        alpha=0.5, squeeze_ratio=2, group_size=2,
        group_num=4, gate_threshold=0.5, group_kernel=3, act=True
    )
    y3 = scconv(x)
    print("SCConv output shape:", y3.shape)                    # [1, 256, 40, 40]