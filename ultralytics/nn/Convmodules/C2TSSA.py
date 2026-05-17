import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class AttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.heads = num_heads

        self.attend = nn.Softmax(dim = 1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )
    
    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h = self.heads)

        b, h, N, d = w.shape
        
        w_normed = torch.nn.functional.normalize(w, dim=-2) 
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp) # b * h * n 
        
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temp'}

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
    
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads    # 注意力头数
        self.head_dim = dim // num_heads  # 每个头的维度
        self.key_dim = int(self.head_dim * attn_ratio)  # key 的通道数
        self.scale = self.key_dim**-0.5  # 分数缩放（避免内积随维度增大而过大）
        nh_kd = self.key_dim * num_heads  # 所有头的 key 总通道数
        h = dim + nh_kd * 2  # Q/K/V 通道总数 = V(dim) + Q(nh_kd) + K(nh_kd)
        self.qkv = Conv(dim, h, 1, act=False)   # 1x1 卷积一次性产生 Q/K/V
        self.proj = Conv(dim, dim, 1, act=False)  # 1x1 卷积做输出投影
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)# 3x3 深度可分离卷积作为位置编码（按通道分组）
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W  # token 数：空间展平后的位置数
        qkv = self.qkv(x)  # 生成拼接的 Q/K/V，形状：[B, h, H, W]，其中 h = dim + 2*num_heads*key_dim

        # 将通道展开为 (num_heads, key_dim/key_dim/head_dim) 三段，并把空间展平为 N
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        # 注意力分数：q^T k → [B, num_heads, N, N]，再做 softmax
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)   # 对每个 query 位置在所有 key 上归一化
        # 将注意力作用到 v 上（这里等价于 V @ Attn^T）
        # 上式右侧第二项为位置编码（对 v 做 depthwise 3x3 卷积后相加，增强空间位置信息）
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)  # 输出投影
        return x


class PSABlock(nn.Module):
    def __init__(self, c: int, attn_ratio: float = 0.5,
                 num_heads: int = 4, shortcut: bool = True) -> None:
        """
        Initialize the PSABlock.

       参数:
            c (int): 输入和输出的通道数。
            attn_ratio (float): 注意力机制中键的维度比例。
            num_heads (int): 多头注意力机制中的头数。
            shortcut (bool): 是否使用shortcut连接。
        """
        super().__init__()

        # 多头注意力子层（位置/空间敏感），输入输出通道均为 c，头数为 num_heads
        self.attn = AttentionTSSA(c, num_heads=num_heads)
        # 前馈网络：1x1 卷积升维到 2c，再用 1x1 卷积降回 c（第二层 act=False 表示无激活）
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        # 是否启用残差（shortcut）连接；True 时执行 x + 子层(x)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 子层1：注意力；若开启残差则 x = x + Attn(x)，否则直接 Attn(x)
        x = x + to_4d(self.attn(to_3d(x)), x.shape[2], x.shape[3]) if self.add else to_4d(self.attn(to_3d(x)), x.shape[2], x.shape[3])
        # 子层2：前馈网络；同样的残差策略
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """
        Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSAN(nn.Module):
    """
    该模块本质上与PSA模块相同，但进行了重构，以允许堆叠更多的PSABlock模块。
    """
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
       参数:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): PSABlock模块的数量。
            e (float): 扩展比例。
        """
        super().__init__()
        assert c1 == c2  # 输入和输出通道数应当相等
        self.c = int(c1 * e)  # 计算隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 第一个1x1卷积，扩展通道数
        self.cv2 = Conv(2 * self.c, c1, 1)  # 第二个1x1卷积，恢复通道数

        # 使用多个PSABlock模块，堆叠n个PSABlock模块
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5,
                                          num_heads=self.c // 128) for _ in range(n)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用cv1卷积层将输入张量分成两个部分
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        # 将b部分通过PSABlock模块进行处理
        b = self.m(b)
        # 将a和处理后的b拼接在一起，并通过cv2卷积层得到输出
        return self.cv2(torch.cat((a, b), 1))
    
if __name__ == "__main__":
    x = torch.randn(1, 256, 40, 40)

    model = C2PSAN(c1=256, c2=256, n=2, e=0.5)
    y = model(x)

    print("input shape :", x.shape)  # [1, 256, 40, 40]
    print("output shape:", y.shape)  # [1, 256, 40, 40]
