import math
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """自动计算 padding，使 stride=1 时尽量保持尺寸不变"""
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


class Rearrange4(nn.Module):
    """
    将 2x2 邻域重排到新的维度
    输入 : [B, C, H, W]
    输出 : [B, C, 4, H/2, W/2]
    """
    def __init__(self):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x):
        b, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, "LAE 要求输入 H、W 可被 2 整除"
        x = self.unshuffle(x)                              # [B, 4C, H/2, W/2]
        x = x.view(b, c, 4, h // 2, w // 2)               # [B, C, 4, H/2, W/2]
        return x


class SharedGroupMap(nn.Module):
    """
    共享的分组 1x1 映射
    输入 : [B, C, 4, H/2, W/2]
    输出 : [B, C_out, 4, H/2, W/2]
    """
    def __init__(self, c1, c2, group_div=16, act=True):
        super().__init__()

        # 先把 [B, C, 4, H, W] 视作 [B, 4C, H, W] 再做分组映射
        in_ch = c1 * 4
        out_ch = c2 * 4

        # 尽量贴近论文中的 “Groups = c/16” 思路，同时保证能整除输入输出通道
        base_g = max(1, c1 // group_div)
        groups = math.gcd(in_ch, out_ch)
        groups = math.gcd(groups, base_g)
        groups = max(1, groups)

        self.map = ConvBNAct(in_ch, out_ch, k=1, s=1, g=groups, act=act)
        self.c2 = c2

    def forward(self, x):
        b, c, n, h, w = x.shape
        x = x.view(b, c * n, h, w)                         # [B, 4C, H/2, W/2]
        x = self.map(x)                                    # [B, 4C_out, H/2, W/2]
        x = x.view(b, self.c2, 4, h, w)                    # [B, C_out, 4, H/2, W/2]
        return x


class LAE(nn.Module):
    """
    LAE: Lightweight Adaptive Extraction
    作用：用于 2x 下采样，同时尽量保留高分辨率细节并进行自适应加权

    参数：
        c1        : 输入通道数
        c2        : 输出通道数
        pool_k    : 自适应分支平均池化核大小
        group_div : 分组卷积的分组控制因子，默认 16
        act       : 是否使用激活函数

    输入：
        x -> [B, c1, H, W]

    输出：
        y -> [B, c2, H/2, W/2]
    """
    def __init__(self, c1, c2, pool_k=3, group_div=16, act=True):
        super().__init__()

        # 轻量重排分支：保留 2x2 邻域信息
        self.rearrange_feat = Rearrange4()

        # 自适应权重分支：先做平均池化，再做同样的重排
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_k, stride=1, padding=pool_k // 2)
        self.rearrange_weight = Rearrange4()

        # 共享分组映射：一套参数同时作用于特征分支和权重分支
        self.shared_map = SharedGroupMap(c1, c2, group_div=group_div, act=act)

        # 归一化 4 个子位置上的权重
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # 输入特征
        # x: [B, c1, H, W]

        # -----------------------------
        # 1) Lightweight Extraction
        #    将 2x2 邻域重排到新的维度
        # -----------------------------
        feat_5d = self.rearrange_feat(x)                   # [B, c1, 4, H/2, W/2]
        feat_5d = self.shared_map(feat_5d)                 # [B, c2, 4, H/2, W/2]

        # -----------------------------
        # 2) Adaptive Extraction
        #    先做局部平均池化，再生成 4 个位置的权重
        # -----------------------------
        weight_in = self.avg_pool(x)                       # [B, c1, H, W]
        weight_5d = self.rearrange_weight(weight_in)       # [B, c1, 4, H/2, W/2]
        weight_5d = self.shared_map(weight_5d)             # [B, c2, 4, H/2, W/2]
        weight_5d = self.softmax(weight_5d)                # [B, c2, 4, H/2, W/2]

        # -----------------------------
        # 3) 自适应加权融合
        #    对 4 个重排子位置做加权求和
        # -----------------------------
        out = (feat_5d * weight_5d).sum(dim=2)             # [B, c2, H/2, W/2]

        return out


if __name__ == "__main__":
    x = torch.randn(1, 256, 80, 80)
    model = LAE(c1=256, c2=256, pool_k=3, group_div=16, act=True)
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 80, 80]
    print("output shape:", y.shape)   # [1, 256, 40, 40]