# Ultralytics YOLO, AGPL-3.0 license
"""LMSFE-YOLO modules extracted from the existing research implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.conv import Conv, RepConv, DWConv
from ..modules.block import C3k, C3k2
from .lga import LGA


class LCSM(nn.Module):
    # Hierarchical Attention Fusion Block
    def __init__(self, inc, ouc, group=False):
        super(LCSM, self).__init__()
        ch_1, ch_2 = inc
        hidc = ouc // 2

        self.lgb1_local = LGA(hidc, 2)
        self.lgb1_global = LGA(hidc, 4)
        self.lgb2_local = LGA(hidc, 2)
        self.lgb2_global = LGA(hidc, 4)

        self.W_x1 = Conv(ch_1, hidc, 1, act=False)
        self.W_x2 = Conv(ch_2, hidc, 1, act=False)
        self.W = Conv(hidc, ouc, 3, g=4)

        self.conv_squeeze = Conv(ouc * 3, ouc, 1)
        self.rep_conv = RepConv(ouc, ouc, 3, g=(16 if group else 1))
        self.conv_final = Conv(ouc, ouc, 1)

    def forward(self, inputs):
        x1, x2 = inputs
        W_x1 = self.W_x1(x1)
        W_x2 = self.W_x2(x2)
        bp = self.W(W_x1 + W_x2)

        x1 = torch.cat([self.lgb1_local(W_x1), self.lgb1_global(W_x1)], dim=1)
        x2 = torch.cat([self.lgb2_local(W_x2), self.lgb2_global(W_x2)], dim=1)

        return self.conv_final(self.rep_conv(self.conv_squeeze(torch.cat([x1, x2, bp], 1))))


class LCSM_NoLocal(LCSM):
    """LCSM ablation with both local-attention branches replaced by identity."""

    def __init__(self, inc, ouc, group=False):
        super().__init__(inc, ouc, group)
        self.lgb1_local = nn.Identity()
        self.lgb2_local = nn.Identity()


class LCSM_NoGlobal(LCSM):
    """LCSM ablation with both global-attention branches replaced by identity."""

    def __init__(self, inc, ouc, group=False):
        super().__init__(inc, ouc, group)
        self.lgb1_global = nn.Identity()
        self.lgb2_global = nn.Identity()


class LCSM_NoLGA(LCSM):
    """LCSM ablation with all LGA branches replaced by identity."""

    def __init__(self, inc, ouc, group=False):
        super().__init__(inc, ouc, group)
        self.lgb1_local = nn.Identity()
        self.lgb1_global = nn.Identity()
        self.lgb2_local = nn.Identity()
        self.lgb2_global = nn.Identity()

__all__ = ['LCSM', 'LCSM_NoGlobal', 'LCSM_NoLGA', 'LCSM_NoLocal']
