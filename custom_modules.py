import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


# -------------------------
# Basic Conv-BN-Act
# -------------------------
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    # For Ultralytics fuse
    def forward_fuse(self, x):
        return self.act(self.conv(x))



# -------------------------
# GhostConv
# -------------------------
class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, ratio=2):
        super().__init__()
        c_ = math.ceil(c2 / ratio)
        self.primary = Conv(c1, c_, k, s)
        self.cheap = Conv(c_, c_, 3, 1, g=c_, act=True)

    def forward(self, x):
        y = self.primary(x)
        z = self.cheap(y)
        out = torch.cat([y, z], dim=1)
        return out[:, : self.primary.conv.out_channels * 2, :, :]


# -------------------------
# C2f (Ultralytics-like)
# -------------------------
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


# -------------------------
# SCCA: Spatial-Channel Collaborative Attention
# -------------------------
class SCCA(nn.Module):
    def __init__(self):
        super().__init__()
        self._built = False

    def _build(self, c1: int, device, dtype):
        c1 = int(c1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c1, max(4, c1 // 16), 1, bias=False).to(device=device, dtype=dtype)
        self.fc2 = nn.Conv2d(max(4, c1 // 16), c1, 1, bias=False).to(device=device, dtype=dtype)
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False).to(device=device, dtype=dtype)
        self._built = True

    def forward(self, x):
        if not self._built:
            self._build(x.shape[1], x.device, x.dtype)

        # channel attention
        y = self.avg_pool(x)
        y = torch.sigmoid(self.fc2(F.silu(self.fc1(y))))
        x = x * y

        # spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        s = torch.sigmoid(self.spatial(torch.cat([max_out, avg_out], dim=1)))
        return x * s


# -------------------------
# MambaBlock2D
# -------------------------
class MambaBlock2D(nn.Module):
    def __init__(self, d_state=16, expand=2):
        super().__init__()
        self.d_state = int(d_state)
        self.expand = int(expand)

        self._built = False
        self.in_proj = None
        self.mamba = None
        self.out_proj = None
        self.norm = None
        self.act = nn.SiLU(inplace=True)

    def _build(self, c1: int, device, dtype):
        c1 = int(c1)
        hidden = c1 * self.expand

        self.in_proj = nn.Conv2d(c1, hidden, 1, bias=False).to(device=device, dtype=dtype)
        self.mamba = Mamba(d_model=hidden, d_state=self.d_state, expand=1).to(device=device, dtype=dtype)
        self.out_proj = nn.Conv2d(hidden, c1, 1, bias=False).to(device=device, dtype=dtype)
        self.norm = nn.BatchNorm2d(c1).to(device=device, dtype=dtype)
        self._built = True

    def forward(self, x):
        # Skip mamba on CPU (e.g. eval/export) to avoid CUDA kernel issues
        if x.device.type != "cuda":
            return x

        if not self._built:
            self._build(int(x.shape[1]), x.device, x.dtype)

        identity = x
        x = self.in_proj(x)

        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        x = self.mamba(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)

        x = self.out_proj(x)
        x = self.norm(x)
        return self.act(x + identity)


# -------------------------
# ADDR: Adaptive Detail-Preserving Downsampling Routine
# Goal: improve small object recall + reduce extreme boundary errors (HD95/ASSD) while staying lightweight
# Usage in YAML:  - [-1, 1, ADDR, []]
# -------------------------
class ADDR(nn.Module):
    """
    ADDR-FG-TrainOnly (final version)
    - Training: feature-guided ambiguity/edge map for lightweight boundary refinement (regularization)
    - Inference: directly return x (zero cost, zero risk, no prediction change)
    - Additional: spatial branch does high-pass enhancement to avoid raising background globally (which causes FP)
    """
    def __init__(self, k=3, reduction=16, warmup_iters=800):
        super().__init__()
        self.k = int(k)
        self.reduction = int(reduction)
        self.warmup_iters = int(warmup_iters)

        # Learn intensity from 0, preserving baseline; gradually learn effective values during training
        self.gamma_s = nn.Parameter(torch.tensor(0.0))
        self.gamma_c = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("_iter", torch.zeros((), dtype=torch.long), persistent=False)

        self._built = False
        self.dw = None
        self.pw = None
        self.bn = None
        self.fc1 = None
        self.fc2 = None
        self.act = nn.SiLU(inplace=True)

    def _build(self, c1: int, device, dtype):
        c1 = int(c1)
        k = self.k
        self.dw = nn.Conv2d(c1, c1, k, padding=k // 2, groups=c1, bias=False).to(device=device, dtype=dtype)
        self.pw = nn.Conv2d(c1, c1, 1, bias=False).to(device=device, dtype=dtype)
        self.bn = nn.BatchNorm2d(c1).to(device=device, dtype=dtype)

        r = max(4, c1 // self.reduction)
        self.fc1 = nn.Conv2d(c1, r, 1, bias=True).to(device=device, dtype=dtype)
        self.fc2 = nn.Conv2d(r, c1, 1, bias=True).to(device=device, dtype=dtype)

        self._built = True

    @staticmethod
    def _edge_strength(x: torch.Tensor) -> torch.Tensor:
        """
        Estimate boundary/ambiguous regions from feature gradient magnitude.
        More stable than using predicted probability p.
        x: (B,C,H,W) -> e: (B,1,H,W) in [0,1]
        """
        f = x.mean(dim=1, keepdim=True)
        gx = f[:, :, :, 1:] - f[:, :, :, :-1]
        gy = f[:, :, 1:, :] - f[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        g = torch.sqrt(gx * gx + gy * gy + 1e-7)

        # per-sample normalize to 0~1
        g_flat = g.flatten(2)
        g_max = g_flat.max(dim=2, keepdim=True)[0].clamp_min(1e-6).unsqueeze(-1)
        return (g / g_max).clamp(0.0, 1.0)

    def forward(self, x):
        if not self._built:
            self._build(int(x.shape[1]), x.device, x.dtype)

        # Key: completely bypass at inference (ADDR does not affect final prediction)
        if not self.training:
            return x

        # Training early warmup: prevent excessive perturbation early on
        self._iter += 1
        if int(self._iter.item()) < self.warmup_iters:
            return x

        # 1) feature-guided ambiguous/edge map (stable, won't fluctuate like p)
        amb = self._edge_strength(x)
        amb = F.max_pool2d(amb, 3, 1, 1)  # continuity

        # 2) channel recalibration (based on global context, does not alter semantic boundaries)
        ctx = x.mean(dim=(2, 3), keepdim=True)
        w = torch.sigmoid(self.fc2(self.act(self.fc1(ctx))))
        x1 = x * (1.0 + self.gamma_c * w)

        # 3) spatial refine: high-pass enhancement to avoid FP expansion
        y = self.bn(self.pw(self.dw(x1)))
        # high-pass: remove low-frequency bias
        y_lp = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        y_hp = y - y_lp
        y_hp = torch.tanh(y_hp)  # limit amplitude

        out = x1 + self.gamma_s * amb * y_hp
        return out
