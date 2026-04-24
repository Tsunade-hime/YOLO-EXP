import torch
import torch.nn as nn


# -----------------------------------------------------------
# 1. YOLO-style basic Conv block
# -----------------------------------------------------------
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

        self.conv = nn.Conv2d(
            c1, c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        # x:   [B, C1, H, W]
        # out: [B, C2, H_out, W_out]
        return self.act(self.bn(self.conv(x)))


# -----------------------------------------------------------
# 2. LSKA block
# -----------------------------------------------------------
class LSKA(nn.Module):
    """
    Large Separable Kernel Attention (engineering-friendly version)

    Core idea:
      - use depth-wise horizontal and vertical large kernels
      - then use 1x1 conv to mix channel responses
      - finally multiply attention map with input feature

    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """
    def __init__(self, dim, k_size=7):
        super().__init__()

        # A practical simplified version for YOLO fusion:
        # DWConv(1, k) -> DWConv(k, 1) -> 1x1 Conv
        self.dw_h = nn.Conv2d(
            dim, dim,
            kernel_size=(1, k_size),
            stride=1,
            padding=(0, k_size // 2),
            groups=dim,
            bias=False
        )
        self.dw_v = nn.Conv2d(
            dim, dim,
            kernel_size=(k_size, 1),
            stride=1,
            padding=(k_size // 2, 0),
            groups=dim,
            bias=False
        )
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        identity = x                         # [B, C, H, W]

        attn = self.dw_h(x)                  # [B, C, H, W]
        attn = self.dw_v(attn)               # [B, C, H, W]
        attn = self.pw(attn)                 # [B, C, H, W]

        # element-wise attention
        out = identity * attn                # [B, C, H, W]
        return out


# -----------------------------------------------------------
# 3. SPPF-LSKA
# -----------------------------------------------------------
class SPPF_LSKA(nn.Module):
    """
    SPPF-LSKA

    Structure:
      1) 1x1 Conv reduce channels
      2) SPPF serial max pooling -> 4-scale concat
      3) LSKA attention refinement
      4) 1x1 Conv output

    This module is designed as a direct replacement for SPPF.

    Input : [B, C1, H, W]
    Output: [B, C2, H, W]

    Recommended YAML style:
      - [-1, 1, SPPF_LSKA, [C2, 5, 7]]

    Constructor keeps SPPF-like style:
      SPPF_LSKA(c1, c2, k=5, lska_k=7)
    """
    def __init__(self, c1, c2, k=5, lska_k=7):
        super().__init__()

        # hidden channels, same design spirit as original SPPF
        c_ = c1 // 2

        # first 1x1 conv: reduce channels
        self.cv1 = Conv(c1, c_, 1, 1)

        # SPPF max pooling
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # LSKA attention is applied on concatenated multi-scale features
        self.lska = LSKA(dim=c_ * 4, k_size=lska_k)

        # final 1x1 conv: fuse 4-scale features to output channels
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

    def forward(self, x):
        """
        x: [B, C1, H, W]
        """
        # -------------------------------------------------
        # Step 1: reduce channels
        # x1: [B, C_, H, W]
        # -------------------------------------------------
        x1 = self.cv1(x)

        # -------------------------------------------------
        # Step 2: SPPF serial max pooling
        # y1: [B, C_, H, W]
        # y2: [B, C_, H, W]
        # y3: [B, C_, H, W]
        # y4: [B, C_, H, W]
        # -------------------------------------------------
        y1 = x1
        y2 = self.m(y1)
        y3 = self.m(y2)
        y4 = self.m(y3)

        # -------------------------------------------------
        # Step 3: concat 4-scale features
        # cat: [B, 4*C_, H, W]
        # -------------------------------------------------
        cat = torch.cat((y1, y2, y3, y4), dim=1)

        # -------------------------------------------------
        # Step 4: LSKA attention refinement
        # cat_lska: [B, 4*C_, H, W]
        # -------------------------------------------------
        cat_lska = self.lska(cat)

        # -------------------------------------------------
        # Step 5: output fusion
        # out: [B, C2, H, W]
        # -------------------------------------------------
        out = self.cv2(cat_lska)
        return out


if __name__ == "__main__":
    # Example
    x = torch.randn(1, 256, 20, 20)
    model = SPPF_LSKA(c1=256, c2=256, k=5, lska_k=7)
    y = model(x)

    print("input shape :", x.shape)   # [1, 256, 20, 20]
    print("output shape:", y.shape)   # [1, 256, 20, 20]