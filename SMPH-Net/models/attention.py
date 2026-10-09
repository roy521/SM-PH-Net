import torch.nn as nn
import torch
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.sigmoid(self.fc(x))
        return x * w


class CrossAttention(nn.Module):
    """Cross attention module.
    Query from one feature, Key/Value from another feature.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Args:
            query: (B, D)
            key: (B, D)
            value: (B, D)
        Returns:
            out: (B, D)
        """
        B, D = query.shape

        q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, D)
        out = self.out_proj(out)

        return out


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention fusion module.
    Implements bidirectional information interaction between two features.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()

        # fa -> fm cross attention (appearance queries motion)
        self.cross_attn_fa2fm = CrossAttention(dim, num_heads, dropout)

        # fm -> fa cross attention (motion queries appearance)
        self.cross_attn_fm2fa = CrossAttention(dim, num_heads, dropout)

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Learnable scaling parameter
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, fa, fm):
        """
        Args:
            fa: (B, D) - Appearance feature
            fm: (B, D) - Motion feature
        Returns:
            fusion: (B, D) - Fused feature
        """
        fa_enhanced = self.cross_attn_fa2fm(fa, fm, fm)
        fm_enhanced = self.cross_attn_fm2fa(fm, fa, fa)

        fa_final = fa + self.alpha * fa_enhanced
        fm_final = fm + (1 - self.alpha) * fm_enhanced

        fusion = torch.cat([fa_final, fm_final], dim=1)
        fusion = self.fusion(fusion)

        return fusion


class GatedFusion(nn.Module):
    """Gated fusion module.
    Uses gating mechanism to adaptively fuse two features.

    Args:
        dim: Feature dimension
    """

    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        self.transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )

    def forward(self, fa, fm):
        """
        Args:
            fa: (B, D) - Appearance feature
            fm: (B, D) - Motion feature
        Returns:
            fusion: (B, D) - Fused feature
        """
        gate_input = torch.cat([fa, fm], dim=1)
        gate_weight = self.gate(gate_input)

        fused = gate_weight * fa + (1 - gate_weight) * fm

        fusion = self.transform(fused)

        return fusion


class SpatialTokenFusion(nn.Module):
    """Spatial token fusion: bidirectional cross-attention on spatial tokens.
    Input A_s, A_m: (B,T,dim,h,w); output A_f: (B,T,dim,h,w).
    """

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn_s = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_m = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_s = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, A_s, A_m):
        B, T, d, h, w = A_s.shape
        N = h * w
        xs = A_s.permute(0, 1, 3, 4, 2).reshape(B * T, N, d)
        xm = A_m.permute(0, 1, 3, 4, 2).reshape(B * T, N, d)

        s2, _ = self.attn_s(xs, xm, xm)
        xs = self.norm_s(xs + s2)
        m2, _ = self.attn_m(xm, xs, xs)
        xm = self.norm_m(xm + m2)

        xf = self.fuse(torch.cat([xs, xm], dim=-1))
        return xf.reshape(B, T, h, w, d).permute(0, 1, 4, 2, 3)


class TemporalPooling(nn.Module):
    """Temporal attention pooling. v_t (B,T,dim) -> v (B,dim)."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, v_t):
        beta = torch.softmax(self.score(v_t), dim=1)
        return (v_t * beta).sum(dim=1)
