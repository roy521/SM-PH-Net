import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.temporal import TemporalSmoothLoss


class RAFTLite(nn.Module):
    """RAFT-Lite: Lightweight optical flow estimation network.
    Input two adjacent MRI frames, output 2-channel flow field (u,v).
    """

    def __init__(self, input_channels=1, base_channels=32):
        super().__init__()

        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Conv2d(input_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Context network
        self.context_net = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Flow prediction head
        self.flow_head = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 2, 3, padding=1)
        )

        # Iterative update count
        self.num_iters = 4

    def forward(self, img1, img2):
        """
        Args:
            img1: First frame (B, 1, H, W)
            img2: Second frame (B, 1, H, W)
        Returns:
            flow: Optical flow field (B, 2, H, W)
        """
        feat1 = self.feature_net(torch.cat([img1, img2], dim=1))
        feat2 = self.context_net(img1)

        flow = torch.zeros(img1.size(0), 2, img1.size(2), img1.size(3),
                          device=img1.device, dtype=img1.dtype)

        for _ in range(self.num_iters):
            grid = self._create_grid(img1.size(2), img1.size(3), img1.device)
            flow_grid = grid + flow.permute(0, 2, 3, 1)
            sampled = F.grid_sample(img2, flow_grid, mode='bilinear',
                                   padding_mode='border', align_corners=True)

            corr_feat = torch.cat([feat1, self.feature_net(torch.cat([img1, sampled], dim=1))], dim=1)

            flow_delta = self.flow_head(torch.cat([corr_feat, feat2], dim=1))

            flow = flow + flow_delta

        return flow

    def _create_grid(self, h, w, device):
        """Create normalized grid."""
        y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                             torch.arange(w, device=device, dtype=torch.float32))
        grid = torch.stack([x, y], dim=-1)
        grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
        grid[..., 0] = 2.0 * grid[..., 0] / (w - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (h - 1) - 1.0
        return grid


class TemporalGaussianSmoothing(nn.Module):
    """Temporal 1D Gaussian smoothing, no trainable parameters.
    Input shape: (B, T, C, H, W)
    Output shape: (B, T, C, H, W)
    """

    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer('kernel', self._create_gaussian_kernel(kernel_size, sigma))

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            Smoothed features (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, 1, T)

        pad_size = self.kernel_size // 2
        x_padded = F.pad(x_flat, (pad_size, pad_size), mode='replicate')

        kernel = self.kernel.to(x.device)

        smoothed = F.conv1d(x_padded, kernel, padding=0)

        smoothed = smoothed.reshape(B, C, H, W, T).permute(0, 4, 1, 2, 3)

        return smoothed

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Create 1D Gaussian kernel."""
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel = torch.exp(-x.pow(2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, -1)


class SpatioTemporalTCNBlock(nn.Module):
    """Spatio-temporal TCN residual block with GroupNorm."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()

        self.t_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(self.t_pad, 0, 0),
            dilation=(dilation, 1, 1)
        )
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout3d(dropout)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(self.t_pad, 0, 0),
            dilation=(dilation, 1, 1)
        )
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout3d(dropout)

        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.final_relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        if self.t_pad > 0:
            out = out[:, :, :-self.t_pad, :, :]
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        if self.t_pad > 0:
            out = out[:, :, :-self.t_pad, :, :]
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        return self.final_relu(out + residual)


class SpatioTemporalTCNEncoder(nn.Module):
    """Spatio-temporal TCN encoder: long-range temporal modeling preserving spatial dimensions.
    Input: (B, in_channels, T, H, W)
    Output: (B, out_channels, T, H, W)
    """

    def __init__(self, in_channels, num_channels=[32, 64], kernel_size=3,
                 dropout=0.2, dilation_factor=2, return_global=False):
        super().__init__()
        self.return_global = return_global
        self.out_channels = num_channels[-1]

        layers = []
        for i in range(len(num_channels)):
            dilation = dilation_factor ** i
            in_ch = in_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(
                SpatioTemporalTCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.tcn_blocks = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, T, H, W)
        Returns:
            (B, out_channels, T, H, W) or (B, out_channels, T) if return_global=True
        """
        out = self.tcn_blocks(x)

        if self.return_global:
            out = torch.mean(out, dim=[3, 4])

        return out


class MotionBranch(nn.Module):
    """Motion branch: RAFT-Lite optical flow + temporal smoothing + TCN encoding."""

    def __init__(self, input_channels=1, flow_channels=2, tcn_channels=[32, 64],
                 smooth_kernel_size=5, smooth_sigma=1.0, smooth_loss_weight=1.0,
                 use_smoothing=True, use_tcn=True):
        super().__init__()

        self.use_smoothing = use_smoothing
        self.use_tcn = use_tcn

        # RAFT-Lite optical flow estimator
        self.flow_estimator = RAFTLite(input_channels=input_channels)

        # Temporal Gaussian smoothing
        self.temporal_smoothing = TemporalGaussianSmoothing(
            kernel_size=smooth_kernel_size,
            sigma=smooth_sigma
        )

        # Spatio-temporal TCN encoder
        self.tcn_encoder = SpatioTemporalTCNEncoder(
            in_channels=flow_channels,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=0.2,
            dilation_factor=2,
            return_global=True
        )

        # Fallback without TCN
        out_ch = tcn_channels[-1]
        self.simple_flow_proj = nn.Linear(flow_channels, out_ch)

        # Temporal smoothness loss
        self.smooth_loss = TemporalSmoothLoss(weight=smooth_loss_weight)

        # Final pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, return_spatial=False):
        """
        Args:
            x: Input sequence (B, T, C, H, W)
            return_spatial: If True, return per-frame spatial feature maps
        Returns:
            feat: Motion feature (B, out_channels) or (B,T,out_ch,H,W)
            flow_loss: Optical flow smoothness loss
        """
        B, T, C, H, W = x.shape

        # Compute adjacent frame optical flows
        flows = []
        for t in range(T - 1):
            frame1 = x[:, t]
            frame2 = x[:, t + 1]
            flow = self.flow_estimator(frame1, frame2)
            flows.append(flow)

        # Stack flow sequence (B, T-1, 2, H, W)
        flow_sequence = torch.stack(flows, dim=1)

        # Compute smoothness loss
        flow_loss = self.smooth_loss(flow_sequence)

        # Temporal Gaussian smoothing
        if self.use_smoothing:
            smoothed_flows = self.temporal_smoothing(flow_sequence)
        else:
            smoothed_flows = flow_sequence

        if return_spatial:
            if not self.use_tcn:
                raise ValueError("return_spatial=True requires use_tcn=True")
            tcn_input = smoothed_flows.permute(0, 2, 1, 3, 4)
            spatial = self.tcn_encoder.tcn_blocks(tcn_input)
            pad = spatial.new_zeros(B, spatial.size(1), 1, H, W)
            spatial = torch.cat([pad, spatial], dim=2)
            return spatial.permute(0, 2, 1, 3, 4), flow_loss

        if self.use_tcn:
            tcn_input = smoothed_flows.permute(0, 2, 1, 3, 4)
            tcn_out = self.tcn_encoder(tcn_input)
            feat = self.pool(tcn_out).squeeze(-1)
        else:
            spatial_mean = smoothed_flows.mean(dim=[3, 4])
            flow_vec = spatial_mean.mean(dim=1)
            feat = self.simple_flow_proj(flow_vec)

        return feat, flow_loss
