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


class FCM(nn.Module):
    """
    FCM
    Feature Complementary Mapping Module

    作用：
        将浅层空间位置信息与深层语义/通道信息做互补映射，
        缓解特征提取过程中的信息不平衡问题。

    参数：
        c1     : 输入通道数
        c2     : 输出通道数
        alpha  : split ratio，控制 3x3 分支所占比例
        act    : 是否使用激活函数

    输入：
        x -> [B, c1, H, W]

    输出：
        y -> [B, c2, H, W]
    """
    def __init__(self, c1, c2, alpha=0.5, act=True):
        super().__init__()
        assert c1 >= 2, "FCM 要求输入通道数 c1 >= 2"
        assert 0.0 < alpha < 1.0, "alpha 必须在 (0, 1) 之间"

        # -----------------------------
        # 1) 通道划分
        #    X1: 走 3x3 分支，提取更强语义/通道信息
        #    X2: 走 1x1 分支，保留更多空间位置信息
        # -----------------------------
        c_sem = max(1, min(int(round(c1 * alpha)), c1 - 1))
        c_spa = c1 - c_sem

        self.c_sem = c_sem
        self.c_spa = c_spa

        # -----------------------------
        # 2) Transformation
        #    X1 -> XC: 3x3 Conv
        #    X2 -> XS: 1x1 Conv
        # -----------------------------
        self.conv_sem = ConvBNAct(c_sem, c2, k=3, s=1, act=act)
        self.conv_spa = ConvBNAct(c_spa, c2, k=1, s=1, act=act)

        # -----------------------------
        # 3) Channel Interaction
        #    DWConv -> GAP -> Sigmoid
        #    从 XC 中生成通道权重 omega1
        # -----------------------------
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.channel_sigmoid = nn.Sigmoid()

        # -----------------------------
        # 4) Spatial Interaction
        #    1x1 Conv -> BN -> Sigmoid
        #    从 XS 中生成空间权重 omega2
        # -----------------------------
        self.spatial_conv = nn.Conv2d(c2, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.spatial_bn = nn.BatchNorm2d(1)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入特征
        # x: [B, c1, H, W]

        # -----------------------------
        # 1) Split
        # -----------------------------
        x1, x2 = torch.split(x, [self.c_sem, self.c_spa], dim=1)
        # x1: [B, c_sem, H, W]
        # x2: [B, c_spa, H, W]

        # -----------------------------
        # 2) Transformation
        # -----------------------------
        xc = self.conv_sem(x1)                              # [B, c2, H, W]
        xs = self.conv_spa(x2)                              # [B, c2, H, W]

        # -----------------------------
        # 3) Channel Interaction
        #    利用 XC 生成通道权重 omega1
        # -----------------------------
        xd = self.dwconv(xc)                                # [B, c2, H, W]
        omega1 = self.avgpool(xd)                           # [B, c2, 1, 1]
        omega1 = self.channel_sigmoid(omega1)               # [B, c2, 1, 1]

        # -----------------------------
        # 4) Spatial Interaction
        #    利用 XS 生成空间权重 omega2
        # -----------------------------
        omega2 = self.spatial_conv(xs)                      # [B, 1, H, W]
        omega2 = self.spatial_bn(omega2)                    # [B, 1, H, W]
        omega2 = self.spatial_sigmoid(omega2)               # [B, 1, H, W]

        # -----------------------------
        # 5) Complementary Mapping + Aggregation
        #    XC 用空间权重增强
        #    XS 用通道权重增强
        # -----------------------------
        xc_map = xc * omega2                                # [B, c2, H, W]
        xs_map = xs * omega1                                # [B, c2, H, W]

        out = xc_map + xs_map                               # [B, c2, H, W]
        return out


if __name__ == "__main__":
    x = torch.randn(1, 256, 40, 40)
    model = FCM(c1=256, c2=256, alpha=0.5, act=True)
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 40, 40]
    print("output shape:", y.shape)   # [1, 256, 40, 40]