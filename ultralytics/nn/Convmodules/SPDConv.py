import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """自动计算 padding，使 stride=1 时尽量保持卷积前后尺寸一致"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvBNAct(nn.Module):
    """基础卷积块：Conv + BN + SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SpaceToDepth(nn.Module):
    """
    Space-to-Depth
    将空间信息重排到通道维

    当 scale=2 时：
    [B, C, H, W] -> [B, 4C, H/2, W/2]
    """
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
        self.op = nn.PixelUnshuffle(scale)

    def forward(self, x):
        b, c, h, w = x.shape
        assert h % self.scale == 0 and w % self.scale == 0, \
            f"Input size {(h, w)} must be divisible by scale={self.scale}"
        return self.op(x)   # [B, C*(scale^2), H/scale, W/scale]


class SPDConv(nn.Module):
    """
    SPD-Conv
    Space-to-Depth Convolution

    作用：
        用 Space-to-Depth + stride=1 Conv 替代 stride>1 的卷积或池化

    参数：
        c1    : 输入通道数
        c2    : 输出通道数
        scale : 下采样倍率，常用 2
        k     : 卷积核大小，常用 3
        act   : 是否使用激活函数

    输入：
        x -> [B, c1, H, W]

    输出：
        y -> [B, c2, H/scale, W/scale]
    """
    def __init__(self, c1, c2, scale=2, k=3, act=True):
        super().__init__()
        self.scale = scale

        # 先做空间到通道的重排：空间尺寸缩小，通道数增大
        self.spd = SpaceToDepth(scale=scale)

        # 重排后通道数变为 c1 * scale^2
        c_mid = c1 * (scale ** 2)

        # 再用 stride=1 卷积进行通道融合与特征提取
        self.conv = ConvBNAct(c_mid, c2, k=k, s=1, act=act)

    def forward(self, x):
        # 输入特征
        # x: [B, c1, H, W]

        x = self.spd(x)           # [B, c1*(scale^2), H/scale, W/scale]
        x = self.conv(x)          # [B, c2, H/scale, W/scale]

        return x


if __name__ == "__main__":
    x = torch.randn(1, 256, 20, 20)
    model = SPDConv(c1=256, c2=256, scale=2, k=3)
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 20, 20]
    print("output shape:", y.shape)   # [1, 256, 10, 10]