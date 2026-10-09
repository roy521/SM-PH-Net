import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone_resnet import ResNet18_MultiScale
from models.motion_branch import MotionBranch
from models.mstf import MSTF
from models.mtab import MTAB
from models.attention import (TemporalAttention, BidirectionalCrossAttention,
                              GatedFusion, SpatialTokenFusion, TemporalPooling)


class DMFNet(nn.Module):
    """
    DMF-Net: Dual-branch Multi-scale Fusion Network with TSM

    Architecture:
        1. ResNet18 backbone extracts multi-scale features (f1, f2, f3)
        2. MTAB module fuses multi-scale features with GatedTSM to get fa
        3. Motion branch extracts motion features fm
        4. Bidirectional cross-attention fuses fa and fm
        5. MLP classifier for PH/Non-PH binary classification
    """

    def __init__(self, fusion_type='cross_attention', num_heads=8, dropout=0.1,
                 use_motion=True, use_temporal_attn=True,
                 use_smoothing=True, use_tcn=True):
        super().__init__()

        self.use_motion = use_motion
        self.use_temporal_attn = use_temporal_attn

        # Backbone: ResNet18 multi-scale feature extraction
        self.backbone = ResNet18_MultiScale()

        # Appearance: GatedTSM + MTAB
        self.mtab = MTAB([64, 128, 256], 128)

        # Motion branch
        if use_motion:
            self.motion_branch = MotionBranch(use_smoothing=use_smoothing,
                                              use_tcn=use_tcn)
            self.fm_proj = nn.Sequential(
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.ReLU()
            )
        else:
            self.motion_branch = None

        # Appearance feature alignment
        self.fa_proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Fusion module
        self.fusion_type = fusion_type
        if not use_motion:
            self.fusion = None
        elif fusion_type == 'cross_attention':
            self.fusion = BidirectionalCrossAttention(128, num_heads, dropout)
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(128)
        else:
            self.fusion = None

        # Temporal attention (optional)
        if use_temporal_attn:
            self.temporal_attention = TemporalAttention(128)
        else:
            self.temporal_attention = None

        # Classifier: PH vs Non-PH binary classification
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input sequence (B, T, C, H, W)
        Returns:
            out: Classification prediction (B, 1)
            flow_loss: Optical flow smoothness loss (0 if no motion branch)
        """
        B, T, C, H, W = x.shape

        # 1. Extract multi-scale features
        x_reshaped = x.view(B * T, C, H, W)
        f1, f2, f3 = self.backbone(x_reshaped)

        # Reshape to sequence format
        f1 = f1.view(B, T, 64, f1.shape[-2], f1.shape[-1])
        f2 = f2.view(B, T, 128, f2.shape[-2], f2.shape[-1])
        f3 = f3.view(B, T, 256, f3.shape[-2], f3.shape[-1])

        # 2. Appearance features fa
        fa = self.mtab(f1, f2, f3)  # (B, 128)
        fa = self.fa_proj(fa)

        # 3. Motion features and fusion
        flow_loss = torch.tensor(0.0, device=x.device)
        if self.use_motion:
            fm, flow_loss = self.motion_branch(x)  # (B, 64), scalar
            fm = self.fm_proj(fm)  # (B, 128)
            if self.fusion_type == 'cross_attention':
                fusion = self.fusion(fa, fm)
            elif self.fusion_type == 'gated':
                fusion = self.fusion(fa, fm)
            else:
                fusion = fa + fm
        else:
            fusion = fa

        # 4. Temporal attention (optional)
        if self.temporal_attention is not None:
            fusion = self.temporal_attention(fusion)

        # 5. Classification prediction
        out = self.classifier(fusion)

        return out, flow_loss
