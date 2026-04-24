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


class RFEMBranch(nn.Module):
    """
    RFEM 单分支：
    1x1 Conv -> 3x3 Dilated Conv

    输入:
        x -> [B, c1, H, W]

    输出:
        y -> [B, c_, H, W]
    """
    def __init__(self, c1, c_, dilation=3):
        super().__init__()
        self.cv1 = ConvBNAct(c1, c_, k=1, s=1)
        self.cv2 = ConvBNAct(c_, c_, k=3, s=1, d=dilation)

    def forward(self, x):
        x = self.cv1(x)      # [B, c_, H, W]
        x = self.cv2(x)      # [B, c_, H, W]
        return x


class RFEM(nn.Module):
    """
    RFEM
    Receptive Field Enhancement Module

    参数:
        c1         : 输入通道数
        c2         : 输出通道数
        dilations  : 空洞率列表，默认 (3, 5, 7)
        e          : 分支隐藏通道比例
        shortcut   : 是否使用残差
        act        : 是否使用激活

    输入:
        x -> [B, c1, H, W]

    输出:
        y -> [B, c2, H, W]
    """
    def __init__(self, c1, c2, dilations=(3, 5, 7), e=0.5, shortcut=True, act=True):
        super().__init__()
        assert len(dilations) >= 1, "dilations 至少需要一个空洞率"

        self.shortcut = shortcut and (c1 == c2)
        self.post_act = nn.SiLU(inplace=True) if act else nn.Identity()

        # shortcut 分支，对应结构图中的顶部 CBS
        self.cv_short = ConvBNAct(c1, c2, k=1, s=1, act=False)

        # 三个分支总的中间通道数
        hidden_total = max(1, int(c2 * e))
        branch_num = len(dilations)
        branch_c = math.ceil(hidden_total / branch_num)

        # 多个不同膨胀率分支
        self.branches = nn.ModuleList([
            RFEMBranch(c1, branch_c, dilation=d) for d in dilations
        ])

        # concat 后再融合到 c2
        self.cv_fuse = ConvBNAct(branch_c * branch_num, c2, k=1, s=1, act=False)

        # 当 c1 != c2 且又不走 shortcut 时，也保证输出能直接返回
        self.cv_identity = ConvBNAct(c1, c2, k=1, s=1, act=False) if (c1 != c2 and not self.shortcut) else nn.Identity()

    def forward(self, x):
        identity = self.cv_short(x) if self.shortcut else self.cv_identity(x)   # [B, c2, H, W]

        # 多分支空洞卷积
        ys = [branch(x) for branch in self.branches]
        # 每个分支输出: [B, branch_c, H, W]

        y = torch.cat(ys, dim=1)                     # [B, branch_c * n, H, W]
        y = self.cv_fuse(y)                          # [B, c2, H, W]

        y = y + identity                             # [B, c2, H, W]
        y = self.post_act(y)                         # [B, c2, H, W]

        return y


if __name__ == "__main__":
    x = torch.randn(1, 256, 40, 40)
    model = RFEM(c1=256, c2=256, dilations=(3, 5, 7), e=0.5, shortcut=True, act=True)
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 40, 40]
    print("output shape:", y.shape)   # [1, 256, 40, 40]