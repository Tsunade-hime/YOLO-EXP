from typing import Tuple
import torch
import torch.nn as nn
try:
    from .SCConv import SCConv
except ImportError:
    from SCConv import SCConv

__all__ = ["C3SCConvExp"]

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """
    标准的卷积模块，包含批量归一化和激活函数。

    属性:
        conv (nn.Conv2d): 卷积层。
        bn (nn.BatchNorm2d): 批量归一化层。
        act (nn.Module): 激活函数层。
        default_act (nn.Module): 默认激活函数（SiLU）。
    """

    default_act = nn.SiLU()  # 默认激活函数是SiLU（Swish）

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        初始化卷积层，使用给定的参数。

        参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): 步幅。
            p (int, 可选): 填充。
            g (int): 分组数。
            d (int): 扩张。
            act (bool | nn.Module): 激活函数（True表示使用默认激活函数，或者提供自定义的激活函数）。
        """
        super().__init__()
        # 初始化卷积层，使用给定的参数
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        # 批量归一化层，作用于卷积的输出
        self.bn = nn.BatchNorm2d(c2)

        # 如果act是True，使用默认激活函数SiLU；如果提供了自定义的激活函数，则使用自定义激活；如果没有激活，则使用nn.Identity()（即不使用激活）
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        将卷积、批量归一化和激活函数应用于输入的张量。

        参数:
            x (torch.Tensor): 输入的张量。

        返回:
            (torch.Tensor): 经过卷积、批量归一化和激活函数处理后的输出张量。
        """
        # 先进行卷积操作，再进行批量归一化，最后应用激活函数
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        仅应用卷积和激活函数，跳过批量归一化。

        参数:
            x (torch.Tensor): 输入的张量。

        返回:
            (torch.Tensor): 经过卷积和激活函数处理后的输出张量，跳过批量归一化。
        """
        # 仅进行卷积和激活函数操作，跳过批量归一化
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True,
                 g: int = 1, e: float = 0.5):
        """
       参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): Bottleneck模块的数量。
            shortcut (bool): 是否使用shortcut连接（跳跃连接）。
            g (int): 卷积操作中的分组数。
            e (float): 扩展比例，决定隐藏通道的数量。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        # 初始化3个卷积层，分别处理输入数据
        self.cv1 = Conv(c1, c_, 1, 1)  # 第一层卷积，1x1卷积
        self.cv2 = Conv(c1, c_, 1, 1)  # 第二层卷积，1x1卷积
        self.cv3 = Conv(2 * c_, c2, 1)  # 第三层卷积，1x1卷积，连接后的结果会通过该卷积层输出
        # 使用多个Bottleneck模块作为CSP Bottleneck的一部分
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0)
                                 for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将cv1和cv2的输出通过m模块处理后合并，最后通过cv3卷积输出
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck_SCConv(nn.Module):
    """用 SCConv 直接替换 Bottleneck 中的 3×3 Conv"""

    def __init__(self, c1: int, c2: int, shortcut: bool = True,
                 g: int = 1,
                 e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        # 第一层仍然保留 1x1 Conv
        self.cv1 = Conv(c1, c_, 1, 1)
        # 直接用 SCConv 替换原来的 3x3 Conv
        self.cv2 = SCConv(c_, c2)
        # shortcut 逻辑保持不变
        self.add = shortcut and c1 == c2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C3SCConvExp(C3):
    """
    C3 + SCConv
    保留 C3 外层框架，仅把内部重复块替换为 Bottleneck_SCConv
    """
    def __init__(self, c1: int, c2: int, n: int = 1,
                 shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        self.m = nn.Sequential(*(
            Bottleneck_SCConv(c_, c_, shortcut, g, e=1.0)
            for _ in range(n)))


if __name__ == "__main__":
    x = torch.randn(1, 256, 40, 40)

    model = C3SCConvExp(c1=256, c2=256, n=2, shortcut=True, g=1, e=0.5)
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 40, 40]
    print("output shape:", y.shape)   # [1, 256, 40, 40]
