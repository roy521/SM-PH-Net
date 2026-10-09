import torch
import torch.nn as nn
import torch.nn.functional as F

from .tsm import GatedTSM
from .attention import TemporalPooling


class TemporalAlign(nn.Module):
    """Temporal alignment: convolution along time dimension (kernel=3 on T, 1 on spatial)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels,
                              kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        B, T, C, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4)
        y = self.conv(y)
        y = self.norm(y)
        y = F.relu(y, inplace=True)
        return y.permute(0, 2, 1, 3, 4)


class SpatialCoAttention(nn.Module):
    """Spatial co-attention: generate 2D spatial attention map from fused features."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 1)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, x):
        attn = torch.sigmoid(self.spatial(x))
        return x * attn


class SEChannelAttention(nn.Module):
    """SE-style channel attention, applied per-frame on (B,T,C,h,w)."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        B, T, C, h, w = feat.shape
        x = feat.reshape(B * T, C, h, w)
        attn = self.se(x).view(B * T, C, 1, 1)
        x = x * attn
        return x.reshape(B, T, C, h, w)


class MTAB(nn.Module):
    """Multiscale Temporal Aggregation Block.

    Args:
        in_channels: List of input channels per scale [64, 128, 256].
        out_channels: Unified output channels (default 128).
        shift_ratio: GatedTSM forward/backward shift channel ratio.
        shift_step: GatedTSM shift step S (frames).
    """

    def __init__(self, in_channels, out_channels=128,
                 shift_ratio=0.5, shift_step=1):
        super().__init__()
        c1, c2, c3 = in_channels
        self.out_channels = out_channels

        # Per-scale residual gated TSM
        self.tsm1 = GatedTSM(c1, shift_ratio=shift_ratio, shift_step=shift_step)
        self.tsm2 = GatedTSM(c2, shift_ratio=shift_ratio, shift_step=shift_step)
        self.tsm3 = GatedTSM(c3, shift_ratio=shift_ratio, shift_step=shift_step)

        # 1x1 conv to unify channels
        self.proj1 = nn.Conv2d(c1, out_channels, 1)
        self.proj2 = nn.Conv2d(c2, out_channels, 1)
        self.proj3 = nn.Conv2d(c3, out_channels, 1)

        # Temporal alignment
        self.t_align1 = TemporalAlign(out_channels)
        self.t_align2 = TemporalAlign(out_channels)
        self.t_align3 = TemporalAlign(out_channels)

        # Concat + Co-attention
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.co_attention = SpatialCoAttention(out_channels)

        # Channel attention
        self.channel_attention = SEChannelAttention(out_channels)

        # Temporal attention pooling
        self.temporal_pool = TemporalPooling(out_channels)

    def _encode(self, f, tsm, proj, t_align):
        """Single scale: GatedTSM -> channel unification -> temporal alignment."""
        B, T, C, h, w = f.shape
        x = tsm(f)
        x = x.reshape(B * T, C, h, w)
        x = proj(x)
        x = x.reshape(B, T, self.out_channels, x.size(-2), x.size(-1))
        x = t_align(x)
        return x

    def aggregate(self, f1, f2, f3):
        """Return fused spatio-temporal map (B,T,out,gh,gw)."""
        x1 = self._encode(f1, self.tsm1, self.proj1, self.t_align1)
        x2 = self._encode(f2, self.tsm2, self.proj2, self.t_align2)
        x3 = self._encode(f3, self.tsm3, self.proj3, self.t_align3)
        B, T = x1.shape[:2]

        # Spatial alignment: unify to deepest scale x3 grid
        gh, gw = x3.shape[-2], x3.shape[-1]

        def space_align(x):
            xt = x.reshape(B * T, self.out_channels, x.size(-2), x.size(-1))
            xt = F.interpolate(xt, size=(gh, gw), mode='bilinear',
                               align_corners=False)
            return xt.reshape(B, T, self.out_channels, gh, gw)

        x1 = space_align(x1)
        x2 = space_align(x2)

        # Concat + Co-attention (per frame)
        frames = []
        for t in range(T):
            cat = torch.cat([x1[:, t], x2[:, t], x3[:, t]], dim=1)
            y = self.fuse(cat)
            y = self.co_attention(y)
            frames.append(y)
        feat = torch.stack(frames, dim=1)

        # Channel attention
        feat = self.channel_attention(feat)
        return feat

    def forward(self, f1, f2, f3):
        """
        Args:
            f1, f2, f3: (B, T, C, h, w)
        Returns:
            fa: (B, out_channels)
        """
        feat = self.aggregate(f1, f2, f3)
        v_t = feat.mean(dim=[3, 4])  # Spatial GAP -> (B,T,out)
        fa = self.temporal_pool(v_t)  # Temporal attention pooling -> (B,out)
        return fa

    def forward_spatial(self, f1, f2, f3):
        """Return fused spatio-temporal map (B,T,out,gh,gw) for Grad-CAM."""
        return self.aggregate(f1, f2, f3)
