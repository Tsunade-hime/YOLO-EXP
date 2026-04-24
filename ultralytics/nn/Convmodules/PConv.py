import math
import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """基础卷积块：Conv + BN + SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, kernel_size=k, stride=s, padding=p,
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AsymPadConv(nn.Module):
    """
    非对称 padding + 定向卷积分支

    参数:
        c1   : 输入通道
        c2   : 输出通道
        k    : 卷积核长度（1×k 或 k×1）
        s    : 步长
        mode : 'h' 表示 1×k，'v' 表示 k×1
        pad  : ZeroPad2d 的 pad，顺序为 (left, right, top, bottom)

    输入:
        x -> [B, c1, H, W]

    输出:
        y -> [B, c2, H/s+1, W/s+1]   # 当 H、W 可被 s 整除时
    """
    def __init__(self, c1, c2, k=3, s=1, mode='h', pad=(0, 0, 0, 0), act=True):
        super().__init__()
        assert mode in ('h', 'v')
        self.pad = nn.ZeroPad2d(pad)

        if mode == 'h':
            kernel = (1, k)
        else:
            kernel = (k, 1)

        self.conv = ConvBNAct(c1, c2, k=kernel, s=s, p=0, act=act)

    def forward(self, x):
        x = self.pad(x)                                   # [B, c1, H+?, W+?]
        x = self.conv(x)                                  # [B, c2, H/s+1, W/s+1]
        return x


class PConv(nn.Module):
    """
    PConv
    Pinwheel-shaped Convolution

    作用：
        用四个方向的非对称卷积分支模拟“风车形”感受野，
        在较小参数增幅下增强底层特征提取与感受野建模能力。

    参数:
        c1   : 输入通道数
        c2   : 输出通道数
        k    : 风车分支卷积长度，常用 3 或 4
        s    : 步长，可替代标准 Conv 的 stride
        act  : 是否使用激活函数

    输入:
        x -> [B, c1, H, W]

    输出:
        y -> [B, c2, H/s, W/s]
    """
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()

        # 四个分支输出通道，工程版允许 c2 不能被 4 整除
        branch_c = math.ceil(c2 / 4)
        self.branch_c = branch_c
        self.s = s
        self.k = k

        # 4 个风车形方向分支
        # pad 顺序: (left, right, top, bottom)
        self.b1 = AsymPadConv(c1, branch_c, k=k, s=s, mode='h', pad=(0, k, 1, 0), act=act)
        self.b2 = AsymPadConv(c1, branch_c, k=k, s=s, mode='v', pad=(0, 1, 0, k), act=act)
        self.b3 = AsymPadConv(c1, branch_c, k=k, s=s, mode='h', pad=(k, 0, 0, 1), act=act)
        self.b4 = AsymPadConv(c1, branch_c, k=k, s=s, mode='v', pad=(1, 0, k, 0), act=act)

        # 分支拼接后，用 2×2 Conv 做融合并恢复到目标输出尺寸
        self.fuse = ConvBNAct(branch_c * 4, c2, k=2, s=1, p=0, act=act)

    def forward(self, x):
        # 输入特征
        # x: [B, c1, H, W]

        y1 = self.b1(x)                                   # [B, c', H/s+1, W/s+1]
        y2 = self.b2(x)                                   # [B, c', H/s+1, W/s+1]
        y3 = self.b3(x)                                   # [B, c', H/s+1, W/s+1]
        y4 = self.b4(x)                                   # [B, c', H/s+1, W/s+1]

        y = torch.cat([y1, y2, y3, y4], dim=1)            # [B, 4c', H/s+1, W/s+1]
        y = self.fuse(y)                                  # [B, c2, H/s, W/s]

        return y


if __name__ == "__main__":
    x = torch.randn(1, 64, 80, 80)

    # 示例1：替代普通 stride=1 卷积
    model1 = PConv(c1=64, c2=64, k=3, s=1, act=True)
    y1 = model1(x)
    print("input shape  :", x.shape)      # [1, 64, 80, 80]
    print("output shape :", y1.shape)     # [1, 64, 80, 80]

    # 示例2：替代 stride=2 下采样卷积
    model2 = PConv(c1=64, c2=128, k=4, s=2, act=True)
    y2 = model2(x)
    print("input shape  :", x.shape)      # [1, 64, 80, 80]
    print("output shape :", y2.shape)     # [1, 128, 40, 40]