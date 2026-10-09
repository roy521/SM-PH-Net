import torch
import torch.nn as nn


class TemporalSmoothLoss(nn.Module):
    """Temporal smoothness regularization loss for optical flow.

    Args:
        weight: Loss weight (default 1.0).
    """

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, flow_sequence):
        """
        Args:
            flow_sequence: Flow sequence (B, T, 2, H, W).
        Returns:
            smooth_loss: Smoothness loss scalar; 0 when T < 2.
        """
        if flow_sequence.size(1) < 2:
            return torch.tensor(0.0, device=flow_sequence.device)

        flow_diff = flow_sequence[:, 1:] - flow_sequence[:, :-1]
        smooth_loss = torch.mean(flow_diff.pow(2))

        return self.weight * smooth_loss
