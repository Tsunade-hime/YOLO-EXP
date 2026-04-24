import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """自动计算 padding，使 stride=1 时尽量保持 H、W 不变"""
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


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：DWConv + PWConv"""
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()

        # 深度卷积：每个通道单独卷积，降低计算量
        self.dw = nn.Conv2d(c1, c1, k, s, autopad(k), groups=c1, bias=False)
        self.dw_bn = nn.BatchNorm2d(c1)
        self.dw_act = nn.SiLU(inplace=True) if act else nn.Identity()

        # 逐点卷积：进行通道融合
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.pw_bn = nn.BatchNorm2d(c2)
        self.pw_act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.dw_act(self.dw_bn(self.dw(x)))   # [B, C, H, W]
        x = self.pw_act(self.pw_bn(self.pw(x)))   # [B, C, H, W]
        return x


class DBSPPF(nn.Module):
    """
    DBSPPF
    Dual-Branch Spatial Pyramid Pooling Fast

    参数:
        c1        : 输入通道数
        c2        : 输出通道数
        k         : 最大池化核大小
        e         : 隐藏通道比例
        dw_kernel : 深度可分离卷积核大小
    """
    def __init__(self, c1, c2, k=5, e=0.5, dw_kernel=3):
        super().__init__()

        # 隐藏层通道数，通常设为输出通道的一半
        c_ = max(1, int(c2 * e))

        # 两个 1×1 Conv：将输入分成上下两个分支
        self.cv_top = ConvBNAct(c1, c_, 1, 1)
        self.cv_bottom = ConvBNAct(c1, c_, 1, 1)

        # 连续池化层：提取多尺度上下文信息
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # 上分支 3 个深度可分离卷积：逐级细化特征
        self.dw1 = DepthwiseSeparableConv(c_, c_, k=dw_kernel, s=1)
        self.dw2 = DepthwiseSeparableConv(c_, c_, k=dw_kernel, s=1)
        self.dw3 = DepthwiseSeparableConv(c_, c_, k=dw_kernel, s=1)

        # 上分支输出压缩
        self.cv_top_out = ConvBNAct(c_, c_, 1, 1)

        # 下分支拼接后压缩：4*c_ -> c_
        self.cv_bottom_out = ConvBNAct(4 * c_, c_, 1, 1)

        # 最终输出融合：2*c_ -> c2
        self.cv_out = ConvBNAct(2 * c_, c2, 1, 1)

    def forward(self, x):
        # 输入特征
        # x: [B, c1, H, W]

        # -----------------------------
        # 1) 双分支初始映射
        # -----------------------------
        top0 = self.cv_top(x)                    # [B, c_, H, W]
        bottom0 = self.cv_bottom(x)              # [B, c_, H, W]

        # -----------------------------
        # 2) 下分支：连续 3 次池化
        #    用于构建多尺度特征
        # -----------------------------
        p1 = self.pool(bottom0)                  # [B, c_, H, W]
        p2 = self.pool(p1)                       # [B, c_, H, W]
        p3 = self.pool(p2)                       # [B, c_, H, W]

        # -----------------------------
        # 3) 上分支：逐级引入池化特征
        #    并通过 DWConv 进行细化
        # -----------------------------
        a1 = top0 + p1                           # 第 1 级融合，   [B, c_, H, W]
        d1 = self.dw1(a1)                        # 第 1 次细化，   [B, c_, H, W]

        a2 = d1 + a1 + p2                        # 第 2 级融合，   [B, c_, H, W]
        d2 = self.dw2(a2)                        # 第 2 次细化，   [B, c_, H, W]

        a3 = d2 + a2 + p3                        # 第 3 级融合，   [B, c_, H, W]
        d3 = self.dw3(a3)                        # 第 3 次细化，   [B, c_, H, W]

        # 与初始上分支特征做逐元素乘法，增强有效响应
        top_out = self.cv_top_out(d3 * top0)     # [B, c_, H, W]

        # -----------------------------
        # 4) 下分支：拼接多尺度池化结果
        # -----------------------------
        bottom_cat = torch.cat([bottom0, p1, p2, p3], dim=1)  # [B, 4*c_, H, W]
        bottom_out = self.cv_bottom_out(bottom_cat)           # [B, c_, H, W]

        # -----------------------------
        # 5) 双分支融合输出
        # -----------------------------
        out = torch.cat([top_out, bottom_out], dim=1)         # [B, 2*c_, H, W]
        out = self.cv_out(out)                                # [B, c2, H, W]

        return out


if __name__ == "__main__":
    x = torch.randn(1, 256, 20, 20)
    model = DBSPPF(c1=256, c2=256, k=5, e=0.5, dw_kernel=3)
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 20, 20]
    print("output shape:", y.shape)   # [1, 256, 20, 20]