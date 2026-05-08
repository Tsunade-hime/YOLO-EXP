import torch
import torch.nn as nn
from typing import Tuple

try:
    from .SCConv import SCConv
except ImportError:
    from SCConv import SCConv

__all__ = ["C3SCTRExp"]

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


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c: int, num_heads: int):
        """
        Initialize a self-attention mechanism using linear transformations and multi-head attention.

        Args:
            c (int): Input and output channel dimension.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a transformer block to the input x and return the output.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after transformer layer.
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """
    Vision Transformer block based on https://arxiv.org/abs/2010.11929.

    This class implements a complete transformer block with optional convolution layer for channel adjustment,
    learnable position embedding, and multiple transformer layers.

    Attributes:
        conv (Conv, optional): Convolution layer if input and output channels differ.
        linear (nn.Linear): Learnable position embedding.
        tr (nn.Sequential): Sequential container of transformer layers.
        c2 (int): Output channel dimension.
    """

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        """
        Initialize a Transformer module with position embedding and specified number of heads and layers.

        Args:
            c1 (int): Input channel dimension.
            c2 (int): Output channel dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagate the input through the transformer block.

        Args:
            x (torch.Tensor): Input tensor with shape [b, c1, w, h].

        Returns:
            (torch.Tensor): Output tensor with shape [b, c2, w, h].
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

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

class C3SCTRExp(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.cv3 = SCConv(2 * c_, c2)  # 第三层卷积，连接后的结果会通过该卷积层输出
        self.m = TransformerBlock(c_, c_, 4, n)

if __name__ == "__main__":
    x = torch.randn(1, 256, 40, 40)

    model = C3SCTRExp(c1=256, c2=256, n=2, shortcut=True, g=1, e=0.5)
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 40, 40]
    print("output shape:", y.shape)   # [1, 256, 40, 40]
