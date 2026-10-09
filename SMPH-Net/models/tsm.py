import torch
import torch.nn as nn


def split_channels(C, shift_ratio):
    """Split channels: return (C_f, C_b, C_k)."""
    c_f = int(C * float(shift_ratio) / 2.0)
    c_b = c_f
    c_k = C - c_f - c_b
    return c_f, c_b, c_k


def shift_features(F, shift_ratio=0.5, shift_step=1):
    """Channel split -> forward/backward shift (step S) -> concat, return Shifted feature F_hat.

    Args:
        F: (B, T, C, H, W)
        shift_ratio: Forward/backward shift channel ratio (C_f, C_b each half).
        shift_step: Shift step S (frames).
    Returns:
        F_hat: (B, T, C, H, W)
    """
    B, T, C, H, W = F.shape
    c_f, c_b, c_k = split_channels(C, shift_ratio)

    F_hat = F.clone()
    S = int(shift_step)

    if T > S and (c_f > 0 or c_b > 0):
        # Forward path: out[t] = in[t-S] (first S frames keep themselves)
        if c_f > 0:
            F_hat[:, S:, 0:c_f] = F[:, :-S, 0:c_f]
        # Backward path: out[t] = in[t+S] (last S frames keep themselves)
        b0, b1 = c_f, c_f + c_b
        if c_b > 0:
            F_hat[:, :-S, b0:b1] = F[:, S:, b0:b1]
        # Keep path: c_k segment unchanged

    return F_hat


class Gate(nn.Module):
    """Gate network: compute per-frame, per-channel gate weights g from F_hat.

    AdaptiveAvgPool2d -> Conv1x1 -> ReLU -> Conv1x1 -> Sigmoid.
    Input (B,T,C,H,W), output g (B,T,C,1,1).
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, F_hat):
        B, T, C, H, W = F_hat.shape
        z = F_hat.reshape(B * T, C, H, W)
        g = self.mlp(self.pool(z))
        return g.view(B, T, C, 1, 1)


class GatedTSM(nn.Module):
    """Residual Gated Temporal Shift Module.

    Args:
        channels: Input channels C.
        shift_ratio: Forward/backward shift channel ratio (default 0.5, each 1/4).
        shift_step: Shift step S (frames, default 1).
        reduction: Gate bottleneck reduction ratio.
    """

    def __init__(self, channels, shift_ratio=0.5, shift_step=1, reduction=8):
        super().__init__()
        self.shift_ratio = shift_ratio
        self.shift_step = shift_step
        self.gate = Gate(channels, reduction)

    def forward(self, F):
        """
        Args:
            F: (B, T, C, H, W)
        Returns:
            F': (B, T, C, H, W), F' = F_hat + F * g
        """
        F_hat = shift_features(F, self.shift_ratio, self.shift_step)
        g = self.gate(F_hat)
        return F_hat + F * g
