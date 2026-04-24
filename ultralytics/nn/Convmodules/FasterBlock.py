import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """自动计算 padding，使 stride=1 时尽量保持特征图尺寸不变"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvBNAct(nn.Module):
    """基础卷积块：Conv + BN + Act"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvBN(nn.Module):
    """基础卷积块：Conv + BN（不带激活）"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.bn(self.conv(x))


class PartialConv(nn.Module):
    """
    Partial Convolution
    仅对一部分通道做 3x3 卷积，其余通道保持不变

    参数:
        c             : 输入/输出通道数（保持不变）
        partial_ratio : 部分卷积通道比例，论文常用 1/4
        k             : 卷积核大小，默认 3

    输入:
        x -> [B, C, H, W]

    输出:
        y -> [B, C, H, W]
    """
    def __init__(self, c, partial_ratio=0.25, k=3):
        super().__init__()
        assert 0 < partial_ratio <= 1.0, "partial_ratio 必须在 (0, 1] 之间"

        self.c_conv = max(1, int(c * partial_ratio))
        self.c_skip = c - self.c_conv

        self.conv = nn.Conv2d(
            self.c_conv, self.c_conv, kernel_size=k, stride=1,
            padding=autopad(k), bias=False
        )

    def forward(self, x):
        x_conv, x_skip = torch.split(x, [self.c_conv, self.c_skip], dim=1)
        x_conv = self.conv(x_conv)                            # [B, c_conv, H, W]
        out = torch.cat((x_conv, x_skip), dim=1)             # [B, C, H, W]
        return out


class FasterBlock(nn.Module):
    """
    FasterBlock
    FasterNet 的基础块：PConv -> PWConv -> BN/ReLU -> PWConv -> Shortcut

    参数:
        c1            : 输入通道数
        c2            : 输出通道数
        expand_ratio  : 中间通道扩张比例
        partial_ratio : PConv 的通道比例
        k             : PConv 卷积核大小
        shortcut      : 是否使用残差
        act           : 是否启用中间激活

    输入:
        x -> [B, c1, H, W]

    输出:
        y -> [B, c2, H, W]
    """
    def __init__(self, c1, c2, expand_ratio=2.0, partial_ratio=0.25, k=3, shortcut=True, act=True):
        super().__init__()

        self.use_shortcut = shortcut and (c1 == c2)

        # 若输入输出通道不一致，先做一次 1x1 对齐，方便后续模块统一处理
        self.cv_in = ConvBN(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()

        hidden = max(1, int(c2 * expand_ratio))

        # 1) PConv：只对部分通道做空间卷积，其余通道直接保留
        self.pconv = PartialConv(c2, partial_ratio=partial_ratio, k=k)

        # 2) 第一层 PWConv：通道扩张
        self.pwconv1 = nn.Conv2d(c2, hidden, kernel_size=1, stride=1, padding=0, bias=False)

        # 3) BN + ReLU：只放在中间层，贴近 FasterNet 论文设计
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

        # 4) 第二层 PWConv：通道压回输出维度
        self.pwconv2 = nn.Conv2d(hidden, c2, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.cv_in(x)                                     # [B, c2, H, W]
        identity = x                                          # [B, c2, H, W]

        x = self.pconv(x)                                     # [B, c2, H, W]
        x = self.pwconv1(x)                                   # [B, hidden, H, W]
        x = self.act(self.bn(x))                              # [B, hidden, H, W]
        x = self.pwconv2(x)                                   # [B, c2, H, W]

        if self.use_shortcut:
            x = x + identity                                  # [B, c2, H, W]

        return x


if __name__ == "__main__":
    x = torch.randn(1, 256, 40, 40)
    model = FasterBlock(c1=256, c2=256, expand_ratio=2.0, partial_ratio=0.25, k=3, shortcut=True, act=True)
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 40, 40]
    print("output shape:", y.shape)   # [1, 256, 40, 40]