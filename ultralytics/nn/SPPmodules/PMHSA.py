import torch
import torch.nn as nn
import torch.nn.functional as F


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


class APDW(nn.Module):
    """
    AP-DW 分支
    Adaptive Pool + DWConv + residual add
    """
    def __init__(self, c, dw_kernel=3):
        super().__init__()
        self.dwconv = nn.Conv2d(
            c, c, kernel_size=dw_kernel, stride=1,
            padding=dw_kernel // 2, groups=c, bias=False
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def deterministic_adaptive_avg_pool2d(self, x, out_size):
        """
        当开启 deterministic 且使用 CUDA 时，
        将 adaptive_avg_pool2d 放到 CPU 上执行，避免触发：
        adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation
        """
        if x.is_cuda and torch.are_deterministic_algorithms_enabled():
            x_cpu = x.float().cpu() if x.dtype in (torch.float16, torch.bfloat16) else x.cpu()
            y_cpu = F.adaptive_avg_pool2d(x_cpu, out_size)
            return y_cpu.to(device=x.device, dtype=x.dtype)
        else:
            return F.adaptive_avg_pool2d(x, out_size)

    def forward(self, x, out_size):
        pooled = self.deterministic_adaptive_avg_pool2d(x, out_size)  # [B, C, Hr, Wr]
        y = pooled + self.act(self.bn(self.dwconv(pooled)))           # [B, C, Hr, Wr]
        return y


class PMHSA(nn.Module):
    """
    P-MHSA
    Pyramid Pooling Multi-Head Self-Attention

    这是一个适配 YOLO 的 2D 特征图版本：
        输入  : [B, c1, H, W]
        输出  : [B, c2, H, W]

    参数:
        c1          : 输入通道数
        c2          : 输出通道数
        num_heads   : 注意力头数
        pool_ratios : 金字塔池化比例，例如 (1, 2, 3, 6)
        qkv_bias    : QKV 是否使用偏置
        attn_drop   : 注意力 dropout
        proj_drop   : 输出投影 dropout
        dw_kernel   : APDW 中 DWConv 的卷积核大小
    """
    def __init__(
        self,
        c1,
        c2,
        num_heads=4,
        pool_ratios=(1, 2, 3, 6),
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        dw_kernel=3
    ):
        super().__init__()
        assert c2 % num_heads == 0, "c2 必须能被 num_heads 整除"

        self.c2 = c2
        self.num_heads = num_heads
        self.head_dim = c2 // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool_ratios = tuple(pool_ratios)

        # 若输入输出通道不同，先用 1×1 Conv 对齐通道
        self.cv_in = ConvBNAct(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # 多个 AP-DW 分支，对应不同池化尺度
        self.apdw_branches = nn.ModuleList([
            APDW(c2, dw_kernel=dw_kernel) for _ in self.pool_ratios
        ])

        # Q 来自原始 token；K/V 来自多尺度池化 token
        self.q = nn.Linear(c2, c2, bias=qkv_bias)
        self.kv = nn.Linear(c2, c2 * 2, bias=qkv_bias)

        # 对拼接后的池化 token 做归一化
        self.norm = nn.LayerNorm(c2)

        # 注意力输出投影
        self.proj = nn.Linear(c2, c2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.cv_in(x)                                      # [B, c2, H, W]
        B, C, H, W = x.shape
        N = H * W

        # -------------------------------------------------
        # 1) 原始特征展平，用于生成 Q
        # -------------------------------------------------
        x_tokens = x.flatten(2).transpose(1, 2)                # [B, N, C]

        q = self.q(x_tokens)                                   # [B, N, C]
        q = q.reshape(B, N, self.num_heads, self.head_dim)     # [B, N, h, d]
        q = q.permute(0, 2, 1, 3)                              # [B, h, N, d]

        # -------------------------------------------------
        # 2) 多尺度 AP-DW 分支，生成 pooled tokens
        # -------------------------------------------------
        pooled_tokens = []
        for ratio, branch in zip(self.pool_ratios, self.apdw_branches):
            Hr = max(1, round(H / ratio))
            Wr = max(1, round(W / ratio))

            pooled_feat = branch(x, (Hr, Wr))                  # [B, C, Hr, Wr]
            pooled_feat = pooled_feat.flatten(2)               # [B, C, Hr*Wr]
            pooled_tokens.append(pooled_feat)

        pooled_tokens = torch.cat(pooled_tokens, dim=2)        # [B, C, M]
        pooled_tokens = pooled_tokens.transpose(1, 2)          # [B, M, C]
        pooled_tokens = self.norm(pooled_tokens)               # [B, M, C]

        # -------------------------------------------------
        # 3) 由 pooled tokens 生成 K 和 V
        # -------------------------------------------------
        kv = self.kv(pooled_tokens)                            # [B, M, 2C]
        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim)  # [B, M, 2, h, d]
        kv = kv.permute(2, 0, 3, 1, 4)                        # [2, B, h, M, d]
        k, v = kv[0], kv[1]                                   # k/v: [B, h, M, d]

        # -------------------------------------------------
        # 4) 多头注意力
        # -------------------------------------------------
        attn = (q @ k.transpose(-2, -1)) * self.scale          # [B, h, N, M]
        attn = attn.softmax(dim=-1)                            # [B, h, N, M]
        attn = self.attn_drop(attn)

        out = attn @ v                                         # [B, h, N, d]
        out = out.transpose(1, 2).contiguous()                 # [B, N, h, d]
        out = out.reshape(B, N, C)                             # [B, N, C]

        # -------------------------------------------------
        # 5) 输出投影，并恢复为 2D 特征图
        # -------------------------------------------------
        out = self.proj(out)                                   # [B, N, C]
        out = self.proj_drop(out)

        out = out.transpose(1, 2).reshape(B, C, H, W)          # [B, C, H, W]
        return out


if __name__ == "__main__":
    x = torch.randn(1, 256, 20, 20)
    model = PMHSA(
        c1=256,
        c2=256,
        num_heads=4,
        pool_ratios=(1, 2, 3, 6),
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        dw_kernel=3
    )
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 20, 20]
    print("output shape:", y.shape)   # [1, 256, 20, 20]