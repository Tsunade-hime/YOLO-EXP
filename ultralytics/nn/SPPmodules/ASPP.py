import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation.*"
)

def autopad(k, p=None, d=1):
    """自动计算 padding，使 stride=1 时尽量保持特征图尺寸不变"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvBNReLU(nn.Module):
    """基础卷积块：Conv + BN + ReLU"""
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SepConvBNReLU(nn.Module):
    """可分离空洞卷积：Depthwise Conv + Pointwise Conv + BN + ReLU"""
    def __init__(self, c1, c2, k=3, s=1, d=1):
        super().__init__()

        # 深度卷积：使用 dilation 扩大感受野
        self.dw = nn.Conv2d(
            c1, c1, kernel_size=k, stride=s,
            padding=autopad(k, d=d), dilation=d,
            groups=c1, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(c1)
        self.dw_act = nn.ReLU(inplace=True)

        # 逐点卷积：完成通道融合
        self.pw = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(c2)
        self.pw_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw_act(self.dw_bn(self.dw(x)))   # [B, C, H, W]
        x = self.pw_act(self.pw_bn(self.pw(x)))   # [B, C, H, W]
        return x


class ASPPPooling(nn.Module):
    """全局池化分支：AdaptiveAvgPool2d -> 1x1 Conv -> BN -> ReLU -> Upsample"""
    def __init__(self, c1, c2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvBNReLU(c1, c2, k=1, s=1)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)                                       # [B, C, 1, 1]
        x = self.conv(x)                                       # [B, c_, 1, 1]
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)  # [B, c_, H, W]
        return x


class ASPP(nn.Module):
    """
    ASPP
    Atrous Spatial Pyramid Pooling

    参数:
        c1            : 输入通道数
        c2            : 输出通道数
        c_            : 每个分支的中间通道数
        dilations     : 空洞率，默认 (12, 24, 36)
        dropout       : 输出 dropout 概率
        use_sep_conv  : 是否使用可分离卷积版本
    """
    def __init__(self, c1, c2, e=0.5, dilations=(12,24,36), dropout=0.1, use_sep_conv=True):
        super().__init__()

        c_ = max(1, int(c2 * e))
        # 1x1 Conv 分支
        self.branch1 = ConvBNReLU(c1, c_, k=1, s=1)

        # 3 个不同空洞率分支
        if use_sep_conv:
            self.branch2 = SepConvBNReLU(c1, c_, k=3, s=1, d=dilations[0])
            self.branch3 = SepConvBNReLU(c1, c_, k=3, s=1, d=dilations[1])
            self.branch4 = SepConvBNReLU(c1, c_, k=3, s=1, d=dilations[2])
        else:
            self.branch2 = ConvBNReLU(c1, c_, k=3, s=1, d=dilations[0])
            self.branch3 = ConvBNReLU(c1, c_, k=3, s=1, d=dilations[1])
            self.branch4 = ConvBNReLU(c1, c_, k=3, s=1, d=dilations[2])

        # 全局池化分支
        self.branch5 = ASPPPooling(c1, c_)

        # 拼接后融合
        self.project = ConvBNReLU(c_ * 5, c2, k=1, s=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入 x: [B, c1, H, W]

        x1 = self.branch1(x)                    # [B, c_, H, W]
        x2 = self.branch2(x)                    # [B, c_, H, W]
        x3 = self.branch3(x)                    # [B, c_, H, W]
        x4 = self.branch4(x)                    # [B, c_, H, W]
        x5 = self.branch5(x)                    # [B, c_, H, W]

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)  # [B, 5*c_, H, W]
        x = self.project(x)                         # [B, c2, H, W]
        x = self.dropout(x)                         # [B, c2, H, W]

        return x


if __name__ == "__main__":
    x = torch.randn(1, 256, 20, 20)

    model = ASPP(
        c1=256,
        c2=256,
        c_=128,
        dilations=(12, 24, 36),
        dropout=0.1,
        use_sep_conv=True
    )

    model.eval()
    with torch.no_grad():
        y = model(x)

    print("input shape :", x.shape)   # [1, 256, 20, 20]
    print("output shape:", y.shape)   # [1, 256, 20, 20]