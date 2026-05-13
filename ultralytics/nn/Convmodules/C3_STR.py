import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import math

try:
    from .Sonic import Sonic
except ImportError:
    from Sonic import Sonic
__all__ = ["C3STR"]


class DFFN_AutoCorr(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DFFN_AutoCorr, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        
        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

        # 自相关融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 控制频域融合强度
        self.beta = nn.Parameter(torch.tensor(0.5))   # 控制空间域融合强度

    def forward(self, x):
        x = self.project_in(x)

        x_patch = rearrange(
            x, 'b c (h ph) (w pw) -> b c h w ph pw',
            ph=self.patch_size, pw=self.patch_size
        )

        # FFT
        Xf = torch.fft.rfft2(x_patch.float())
        Xf = Xf * self.fft
        # 自相关功率谱
        power = Xf * torch.conj(Xf)          # |X|^2
        R = torch.fft.irfft2(power, s=(self.patch_size, self.patch_size))

        # 融合（频域 + 空间域）
        Xf_new = Xf + self.alpha * power     # 频域增强周期结构
        x_patch_new = torch.fft.irfft2(Xf_new, s=(self.patch_size, self.patch_size))
        x_patch_new = x_patch_new + self.beta * R  # 空间域增强

        # 重组
        x = rearrange(
            x_patch_new, 'b c h w ph pw -> b c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size
        )

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
    def forward(self, x):
        # ---- 新增：尺寸适配 ----
        B, C, H, W = x.shape
        ph, pw = self.patch_size, self.patch_size
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # 在右侧和底部做零填充
        # ---- 原流程 ----
        x = self.project_in(x)

        x_patch = rearrange(
            x, 'b c (h ph) (w pw) -> b c h w ph pw',
            ph=self.patch_size, pw=self.patch_size
        )

        # FFT
        Xf = torch.fft.rfft2(x_patch.float())
        Xf = Xf * self.fft
        # 自相关功率谱
        power = Xf * torch.conj(Xf)          # |X|^2
        R = torch.fft.irfft2(power, s=(self.patch_size, self.patch_size))

        # 融合（频域 + 空间域）
        Xf_new = Xf + self.alpha * power     # 频域增强周期结构
        x_patch_new = torch.fft.irfft2(Xf_new, s=(self.patch_size, self.patch_size))
        x_patch_new = x_patch_new + self.beta * R  # 空间域增强

        # 重组
        x = rearrange(
            x_patch_new, 'b c h w ph pw -> b c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size
        )

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        # ---- 新增：裁剪回原始尺寸 ----
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x

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
    def __init__(self, c: int, num_heads: int, ffn_expansion_factor: float = 2.0):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        # 用 DFFN_AutoCorr 替换原来的 fc1 + fc2
        self.ffn = DFFN_AutoCorr(dim=c, ffn_expansion_factor=ffn_expansion_factor, bias=False)

    def forward(self, x: torch.Tensor, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x: 输入序列，形状为 (L, B, C)
            spatial_shape: (H, W)，即 2D 特征图的高和宽
        Returns:
            (L, B, C)
        """
        # 1. 多头自注意力（仍在序列上操作）
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x

        # 2. 转换为二维特征图 → DFFN → 再转回序列
        B = x.shape[1]
        H, W = spatial_shape
        # (L, B, C) -> (B, C, H, W)
        x_2d = x.permute(1, 2, 0).view(B, -1, H, W)   # 假设 L = H*W
        x_2d = self.ffn(x_2d) + x_2d                   # 残差连接（DFFN 内部无残差，这里加更安全）
        # 再转回序列
        x = x_2d.view(B, -1, H*W).permute(2, 0, 1)     # (L, B, C)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)          # 可学习的位置编码
        self.tr = nn.Sequential(
            *(TransformerLayer(c2, num_heads) for _ in range(num_layers))
        )
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, c1, W, H)   # 注意：原代码是 [b, c1, w, h]
        Returns: (B, c2, W, H)
        """
        if self.conv is not None:
            x = self.conv(x)                     # (B, c2, W, H)
        B, C, W, H = x.shape                     # 改称 C 更清晰
        # 展平空间维度并转成 (L, B, C)
        p = x.flatten(2).permute(2, 0, 1)        # (L, B, C)
        # 加上位置编码
        p = p + self.linear(p)
        # 逐层 Transformer（需要 spatial_shape）
        for layer in self.tr:
            p = layer(p, (H, W))                 # 传入 (H, W)
        # 恢复为 (B, C, W, H)
        return p.permute(1, 2, 0).reshape(B, C, W, H)

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

class C3STR(C3):
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
        self.cv1 = Sonic(c1, c_)
        self.cv2 = Sonic(c1, c_)
        self.cv3 = Sonic(2 * c_, c2)
        self.m = TransformerBlock(c_, c_, 4, n)

if __name__ == "__main__":
    x = torch.randn(1, 256, 40, 40)

    model = C3STR(c1=256, c2=256, n=2, shortcut=True, g=1, e=0.5)
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 40, 40]
    print("output shape:", y.shape)   # [1, 256, 40, 40]
