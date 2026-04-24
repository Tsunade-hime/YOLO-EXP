import math
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


class DWConvBNAct(nn.Module):
    """深度卷积块：DWConv + BN + SiLU"""
    def __init__(self, c, k=3, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c, c, k, s, autopad(k),
            groups=c, bias=False
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MSBranchBlock(nn.Module):
    """
    MSBlock 中单个分支的轻量卷积模块
    这里采用 YOLO 适配版 IBM 风格：
        1x1 expand -> DWConv -> 1x1 project

    输入:
        x -> [B, c, H, W]
    输出:
        y -> [B, c, H, W]
    """
    def __init__(self, c, k=3, expand_ratio=2.0, act=True):
        super().__init__()
        hidden = max(1, int(c * expand_ratio))

        self.cv1 = ConvBNAct(c, hidden, k=1, s=1, act=act)
        self.dw = DWConvBNAct(hidden, k=k, s=1, act=act)
        self.cv2 = ConvBNAct(hidden, c, k=1, s=1, act=act)

    def forward(self, x):
        x = self.cv1(x)   # [B, hidden, H, W]
        x = self.dw(x)    # [B, hidden, H, W]
        x = self.cv2(x)   # [B, c, H, W]
        return x


class MSBlock(nn.Module):
    """
    MSBlock
    Multi-Scale Block

    参数:
        c1           : 输入通道数
        c2           : 输出通道数
        branches     : 分支数，论文默认推荐 3
        k            : 分支中的卷积核大小
        expand_ratio : 分支内部扩张比例
        shortcut     : 是否使用残差
        act          : 是否使用激活函数

    输入:
        x -> [B, c1, H, W]

    输出:
        y -> [B, c2, H, W]
    """
    def __init__(self, c1, c2, branches=3, k=3, expand_ratio=2.0, shortcut=False, act=True):
        super().__init__()
        assert branches >= 2, "MSBlock 的 branches 至少为 2"

        self.branches = branches
        self.shortcut = shortcut and (c1 == c2)

        # 为了方便均匀 split，这里先将总通道映射到 branch_channels * branches
        branch_channels = math.ceil(c2 / branches)
        hidden_total = branch_channels * branches

        # 1x1 Conv：先做通道变换，为后续 split 做准备
        self.cv1 = ConvBNAct(c1, hidden_total, k=1, s=1, act=act)

        # 分支模块：第 1 个分支直接保留，其余分支做层级式传播
        self.blocks = nn.ModuleList([
            MSBranchBlock(branch_channels, k=k, expand_ratio=expand_ratio, act=act)
            for _ in range(branches - 1)
        ])

        # concat 后再用 1x1 Conv 融合，并压到目标输出通道 c2
        self.cv2 = ConvBNAct(hidden_total, c2, k=1, s=1, act=act)

    def forward(self, x):
        identity = x                                        # [B, c1, H, W]

        # -----------------------------
        # 1) 先用 1x1 Conv 做通道映射
        # -----------------------------
        x = self.cv1(x)                                     # [B, hidden_total, H, W]

        # -----------------------------
        # 2) 按通道维 split 成多个分支
        # -----------------------------
        xs = torch.chunk(x, self.branches, dim=1)
        # 每个分支: [B, branch_channels, H, W]

        ys = [xs[0]]                                        # 第 1 个分支直接保留

        # -----------------------------
        # 3) 层级式分支传播
        #    Yi = F(Yi-1 + Xi), i > 1
        # -----------------------------
        for i in range(1, self.branches):
            cur = ys[i - 1] + xs[i]                         # [B, branch_channels, H, W]
            cur = self.blocks[i - 1](cur)                   # [B, branch_channels, H, W]
            ys.append(cur)

        # -----------------------------
        # 4) 拼接所有分支，再用 1x1 Conv 融合
        # -----------------------------
        out = torch.cat(ys, dim=1)                          # [B, hidden_total, H, W]
        out = self.cv2(out)                                 # [B, c2, H, W]

        # -----------------------------
        # 5) 可选残差连接
        # -----------------------------
        if self.shortcut:
            out = out + identity                            # [B, c2, H, W]

        return out


if __name__ == "__main__":
    x = torch.randn(1, 256, 40, 40)
    model = MSBlock(c1=256, c2=256, branches=3, k=3, expand_ratio=2.0, shortcut=True)
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 40, 40]
    print("output shape:", y.shape)   # [1, 256, 40, 40]