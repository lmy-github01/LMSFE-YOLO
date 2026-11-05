# Ultralytics YOLO, AGPL-3.0 license
"""LMSFE-YOLO modules extracted from the existing research implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.conv import Conv, RepConv, DWConv
from ..modules.block import C3k, C3k2


class EFE(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class MSEE(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()
        
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EFE(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.final_conv = Conv(inc * 2, inc)
    
    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(torch.cat(out, 1))


class C3k_MSEE(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MSEE(c_, [3, 6, 9, 12]) for _ in range(n)))


class C3k2_MSEE(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MSEE(self.c, self.c, 2, shortcut, g) if c3k else MSEE(self.c, [3, 6, 9, 12]) for _ in range(n))


class _ZeroFeature(nn.Module):
    """Return a zero feature map while preserving requested output channels."""

    def __init__(self, out_channels=None):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        if self.out_channels is None:
            return torch.zeros_like(x)
        return x.new_zeros((x.shape[0], self.out_channels, x.shape[2], x.shape[3]))


def _get_msee_blocks(module):
    return [m for m in module.modules() if isinstance(m, MSEE)]


class C3k2_MSEE_NoEFE(C3k2_MSEE):
    """Ablate EFE while retaining all multi-scale feature branches."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        for block in _get_msee_blocks(self):
            block.ees = nn.ModuleList(nn.Identity() for _ in block.ees)


class C3k2_MSEE_NoMultiScale(C3k2_MSEE):
    """Ablate pooled pyramid branches while retaining the local convolution branch."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        for block in _get_msee_blocks(self):
            branch_channels = block.final_conv.conv.in_channels // (2 * len(block.features))
            block.features = nn.ModuleList(_ZeroFeature(branch_channels) for _ in block.features)
            block.ees = nn.ModuleList(nn.Identity() for _ in block.ees)


class C3k2_MSEE_NoLocalConv(C3k2_MSEE):
    """Ablate the local convolution branch while retaining pyramid edge branches."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        for block in _get_msee_blocks(self):
            block.local_conv = _ZeroFeature()

__all__ = ['C3k2_MSEE', 'C3k2_MSEE_NoEFE', 'C3k2_MSEE_NoLocalConv', 'C3k2_MSEE_NoMultiScale', 'C3k_MSEE', 'EFE', 'MSEE']
