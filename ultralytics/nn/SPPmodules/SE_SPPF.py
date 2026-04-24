import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1. YOLO-style basic Conv
# -----------------------------
class Conv(nn.Module):
    """
    Standard Conv-BN-SiLU block

    Input : [B, C1, H, W]
    Output: [B, C2, H_out, W_out]
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, tuple):
                p = tuple(x // 2 for x in k)
            else:
                p = k // 2
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        # x: [B, C1, H, W]
        # out: [B, C2, H_out, W_out]
        return self.act(self.bn(self.conv(x)))


# -----------------------------
# 2. SENetV2-like channel weighting block
# -----------------------------
class SENetV2Lite(nn.Module):
    """
    Engineering-friendly SENetV2-like block

    Paper-aligned idea:
      - multi-branch pooled descriptors
      - each branch uses small conv projection
      - concat -> fuse conv -> sigmoid
      - reweight input feature map

    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """
    def __init__(self, channels, reduction=16, branches=4):
        super().__init__()
        assert branches >= 1
        hidden = max(channels // reduction, 8)

        # Multi-branch pooled descriptors
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),        # [B, C, H, W] -> [B, C, 1, 1]
                nn.Conv2d(channels, hidden, 1, 1, 0, bias=True),
                nn.ReLU(inplace=True)
            )
            for _ in range(branches)
        ])

        # Fuse branch descriptors back to channel weights
        self.fuse = nn.Conv2d(hidden * branches, channels, 1, 1, 0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        identity = x

        outs = []
        for branch in self.branches:
            # each branch output: [B, hidden, 1, 1]
            outs.append(branch(x))

        # concat: [B, hidden*branches, 1, 1]
        y = torch.cat(outs, dim=1)

        # fuse to channel weights: [B, C, 1, 1]
        y = self.fuse(y)
        y = self.sigmoid(y)

        # channel reweight: [B, C, H, W]
        return identity * y


# -----------------------------
# 3. SE-SPPF
# -----------------------------
class SE_SPPF(nn.Module):
    """
    SE-SPPF: SENetV2 + SPPF-style multi-scale pooling + fusion

    Recommended as a direct replacement for SPPF.

    Input : [B, C1, H, W]
    Output: [B, C2, H, W]

    Structure:
      1) channel weighting by SENetV2-like block
      2) 1x1 conv to hidden channels
      3) SPPF pooling (x, m(x), m(m(x)), m(m(m(x))))
      4) concat pooled maps -> 1x1 fuse
      5) concat with weighted feature map
      6) 1x1 conv + 3x3 conv refine
    """
    def __init__(self, c1, c2, k=5, reduction=16, branches=4):
        super().__init__()

        # hidden channels, aligned with SPPF design
        c_ = c1 // 2

        # (1) channel weighting
        self.se = SENetV2Lite(c1, reduction=reduction, branches=branches)

        # (2) adjust channels before SPPF
        self.cv1 = Conv(c1, c_, 1, 1)

        # (3) SPPF maxpool
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # (4) fuse the 4 pooled maps
        self.cv2 = Conv(c_ * 4, c_, 1, 1)

        # (5) fuse weighted feature map + pooled feature
        self.cv3 = Conv(c1 + c_, c_, 1, 1)

        # (6) final refinement
        self.cv4 = Conv(c_, c2, 3, 1)

    def forward(self, x):
        """
        x: [B, C1, H, W]
        """
        # -----------------------------------
        # Step 1: channel-weighted feature map
        # xw: [B, C1, H, W]
        # -----------------------------------
        xw = self.se(x)

        # -----------------------------------
        # Step 2: reduce channels before SPPF
        # x1: [B, C_, H, W]
        # -----------------------------------
        x1 = self.cv1(xw)

        # -----------------------------------
        # Step 3: serial max pooling (SPPF)
        # y1: [B, C_, H, W]
        # y2: [B, C_, H, W]
        # y3: [B, C_, H, W]
        # y4: [B, C_, H, W]
        # -----------------------------------
        y1 = x1
        y2 = self.m(y1)
        y3 = self.m(y2)
        y4 = self.m(y3)

        # -----------------------------------
        # Step 4: concat pooled multi-scale features
        # cat_sppf: [B, 4*C_, H, W]
        # sppf_out: [B, C_, H, W]
        # -----------------------------------
        cat_sppf = torch.cat([y1, y2, y3, y4], dim=1)
        sppf_out = self.cv2(cat_sppf)

        # -----------------------------------
        # Step 5: concat weighted input feature + sppf feature
        # cat_fuse: [B, C1 + C_, H, W]
        # fuse_out: [B, C_, H, W]
        # -----------------------------------
        cat_fuse = torch.cat([xw, sppf_out], dim=1)
        fuse_out = self.cv3(cat_fuse)

        # -----------------------------------
        # Step 6: final 3x3 refine
        # out: [B, C2, H, W]
        # -----------------------------------
        out = self.cv4(fuse_out)

        return out


if __name__ == "__main__":
    x = torch.randn(1, 256, 20, 20)
    model = SE_SPPF(c1=256, c2=256, k=5, reduction=16, branches=4)
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 20, 20]
    print("output shape:", y.shape)   # [1, 256, 20, 20]