import torch
import torch.nn as nn
import torch.nn.functional as F


class MSTF(nn.Module):
    """Multi-Scale Feature Fusion module.

    Args:
        in_channels: List of input channels per scale [64, 128, 256].
        out_channels: Output channels.
        use_attention: Whether to use channel attention.
    """

    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()

        # 1x1 conv dimension reduction per scale
        self.conv1 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels[2], out_channels, 1)

        # Feature fusion layer
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Channel attention (optional)
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, out_channels // 4),
                nn.ReLU(),
                nn.Linear(out_channels // 4, out_channels),
                nn.Sigmoid()
            )

    def forward(self, f1, f2, f3):
        """
        Args:
            f1: layer1 output (B, T, 64, H, W)
            f2: layer2 output (B, T, 128, H, W)
            f3: layer3 output (B, T, 256, H, W)
        Returns:
            fa: Fused feature (B, out_channels)
        """
        # Take last frame
        f1 = f1[:, -1]
        f2 = f2[:, -1]
        f3 = f3[:, -1]

        # Dimension reduction and upsample to same size
        f1 = self.conv1(f1)
        f2 = F.interpolate(self.conv2(f2), size=f1.shape[-2:])
        f3 = F.interpolate(self.conv3(f3), size=f1.shape[-2:])

        # Concatenate and fuse
        feat = torch.cat([f1, f2, f3], dim=1)
        feat = self.fuse(feat)

        # Channel attention
        if self.use_attention:
            attn = self.channel_attention(feat)
            attn = attn.unsqueeze(-1).unsqueeze(-1)
            feat = feat * attn

        # Global average pooling
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)

        return feat

    def forward_spatial(self, f1, f2, f3):
        """Per-frame multi-scale fusion preserving spatial grid.
        Args:
            f1, f2, f3: (B, T, C, h, w)
        Returns:
            A_s: (B, T, out_channels, h3, w3)
        """
        B, T = f1.shape[:2]
        f1 = f1.reshape(B * T, *f1.shape[2:])
        f2 = f2.reshape(B * T, *f2.shape[2:])
        f3 = f3.reshape(B * T, *f3.shape[2:])

        x1 = self.conv1(f1)
        x2 = self.conv2(f2)
        x3 = self.conv3(f3)
        grid = (x3.shape[-2], x3.shape[-1])
        x1 = F.interpolate(x1, size=grid)
        x2 = F.interpolate(x2, size=grid)

        feat = self.fuse(torch.cat([x1, x2, x3], dim=1))
        if self.use_attention:
            attn = self.channel_attention(feat).unsqueeze(-1).unsqueeze(-1)
            feat = feat * attn
        return feat.reshape(B, T, feat.size(1), grid[0], grid[1])
