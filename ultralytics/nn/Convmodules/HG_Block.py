import math
import torch
import torch.nn as nn


__all__ = ["HGBlockExp"]


def autopad(k, p=None, d=1):
    """自动计算 padding，使 stride=1 时尽量保持特征图尺寸不变"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def _get_act(act=True):
    """统一解析激活函数，HGBlock 默认更贴近 PPHGNetV2 风格，使用 ReLU"""
    if act is True:
        return nn.ReLU(inplace=True)
    if act is False:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    return nn.ReLU(inplace=True)


class _HGBaseConv(nn.Module):
    """
    基础卷积模块：Conv + BN + Act
    仅供当前文件内部使用，不建议直接对外暴露。
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = _get_act(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))   # [B, c2, H, W]


class _HGBaseDWConv(_HGBaseConv):
    """
    深度卷积形式
    groups 取输入输出通道数最大公约数
    """
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class _HGBaseLightConv(nn.Module):
    """
    LightConv

    结构：
        1x1 Conv -> DWConv
    """
    def __init__(self, c1, c2, k=1, act=True):
        super().__init__()
        self.conv1 = _HGBaseConv(c1, c2, 1, 1, act=False)    # [B, c1, H, W] -> [B, c2, H, W]
        self.conv2 = _HGBaseDWConv(c2, c2, k, 1, act=act)    # [B, c2, H, W] -> [B, c2, H, W]

    def forward(self, x):
        return self.conv2(self.conv1(x))                      # [B, c2, H, W]


class HGBlockExp(nn.Module):
    """
    HGBlockExp

    说明：
        这是实验版 HGBlock 模块。
        它继承了 PPHGNetV2 风格 HGBlock 的核心结构：
        重复 LightConv/Conv -> 多层输出拼接 -> squeeze/excitation -> shortcut(optional)

    参数：
        c1        : 输入通道数
        c2        : 输出通道数
        n         : 内部 LightConv/Conv 的重复次数
        cm        : 中间通道数；若为 None，则默认取 c2 // 2
        k         : 卷积核大小
        lightconv : 是否使用 LightConv；False 时改用普通 Conv
        shortcut  : 是否使用 shortcut
        act       : 激活函数；True 时默认使用 ReLU
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 6,
        cm: int = None,
        k: int = 3,
        lightconv: bool = False,
        shortcut: bool = False,
        act=True,
    ):
        super().__init__()

        cm = max(1, c2 // 2) if cm is None else cm           # 中间通道数
        block = _HGBaseLightConv if lightconv else _HGBaseConv

        # 主体重复块
        self.m = nn.ModuleList(
            block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n)
        )

        # squeeze conv + excitation conv
        self.sc = _HGBaseConv(c1 + n * cm, c2 // 2, 1, 1, act=act)   # [B, c1+n*cm, H, W] -> [B, c2/2, H, W]
        self.ec = _HGBaseConv(c2 // 2, c2, 1, 1, act=act)            # [B, c2/2, H, W] -> [B, c2, H, W]

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [x]                                                      # 第 0 项保留输入特征 [B, c1, H, W]
        y.extend(m(y[-1]) for m in self.m)                           # 逐层堆叠，每项 [B, cm, H, W]
        y = torch.cat(y, 1)                                          # [B, c1+n*cm, H, W]
        y = self.ec(self.sc(y))                                      # [B, c2, H, W]
        return y + x if self.add else y                              # [B, c2, H, W]


if __name__ == "__main__":
    x = torch.randn(1, 256, 40, 40)

    # 普通 Conv 版本
    model1 = HGBlockExp(
        c1=256,
        c2=256,
        n=6,
        cm=128,
        k=3,
        lightconv=False,
        shortcut=True,
        act=True,
    )
    y1 = model1(x)
    print("HGBlockExp(Conv) input shape :", x.shape)   # [1, 256, 40, 40]
    print("HGBlockExp(Conv) output shape:", y1.shape)  # [1, 256, 40, 40]

    # LightConv 版本
    model2 = HGBlockExp(
        c1=256,
        c2=256,
        n=6,
        cm=128,
        k=3,
        lightconv=True,
        shortcut=True,
        act=True,
    )
    y2 = model2(x)
    print("HGBlockExp(LightConv) input shape :", x.shape)   # [1, 256, 40, 40]
    print("HGBlockExp(LightConv) output shape:", y2.shape)  # [1, 256, 40, 40]