"""RDAH-Net official architecture — VERBATIM port (backend: ``rdah``).

Provenance (do not diverge — the released checkpoints must strict-load):
    Repository : https://github.com/Elenairene/RDAH-Net  (single commit 373bca2)
    Paper      : "RDAH-Net: Bridging Relative Depth and Absolute Height for
                 Monocular Height Estimation in Remote Sensing",
                 Remote Sens. 2026, 18(7), 1024 —
                 https://www.mdpi.com/2072-4292/18/7/1024
    Classes copied from the official ``test.py`` (lines 17-441):
        ConvLayer, BlockAttention, MobileViTBlock, MobileViT_S_Light,
        CBAM, PositionalEncoding, LightCrossAttention, LightTransformerBlock,
        HeightPredTransformer
    Only changes: module docstring, English comments, ``from __future__``
    import, type hints on the wrapper — NO architectural edits. Every layer
    name, kernel, stride, head count, block size, activation and buffer
    registration matches the official implementation bit-for-bit so that
    ``load_state_dict(..., strict=True)`` succeeds on the released
    checkpoints (verified on the Track1 checkpoint: 0 missing / 0 unexpected).

Contract (derived from the official code + paper, empirically verified
against the released Track1 checkpoint — see depthwizard/rdah.py):
    forward(depth, img):
        depth : [B, 1, H, W] float32 — RAW Depth-Anything-V2 relative depth
                multiplied by RDAH_DEPTH_SCALE (default 40.0), i.e. values in
                the ~[0, 255] range. NOT divided by 255, NOT min-max
                normalized, NOT ImageNet-normalized.
        img   : [B, 3, H, W] float32 — uint8 RGB / 255, then ImageNet
                mean/std normalization (exactly once, by the caller).
        return: [B, 1, H, W] float32 — predicted nDSM height in METRES
                (paper: "absolute height ... with meter units"; checkpoint
                val SmoothL1 loss 1.205 is metre-consistent).

Resolution constraints (architecture-enforced):
    * H, W must be divisible by 128 (BlockAttention reshapes each encoder
      scale — H/4, H/8, H/16 — into 8x8 blocks).
    * H, W <= 1024 for a 1024-trained checkpoint (PositionalEncoding buffer
      is a fixed 64x64 = input/16 grid; larger inputs cannot slice it).
    The official training resolutions: 1024x1024 for DFC2019-Track1,
    512x512 for Swiss/HK (paper Sec. "Input preprocessing and normalization").
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """MobileViT base conv block (Conv+BN+activation) [official: ConvLayer]."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BlockAttention(nn.Module):
    """Blocked window attention [official: BlockAttention]."""

    def __init__(self, dim, num_heads=4, block_size=8, mlp_dim=None, dropout=0.):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim={dim} must be an integer multiple of num_heads={num_heads} "
        )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.mlp_dim = mlp_dim or dim * 2
        self.scale = self.head_dim ** -0.5

        self.local_proj = ConvLayer(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, self.mlp_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(self.mlp_dim, dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        local_feat = self.local_proj(x)

        num_blocks_h = H // self.block_size
        num_blocks_w = W // self.block_size
        num_blocks = num_blocks_h * num_blocks_w

        x_blocked = x.reshape(
            B, C, num_blocks_h, self.block_size, num_blocks_w, self.block_size
        ).permute(0, 2, 4, 1, 3, 5)
        x_blocked = x_blocked.reshape(
            B * num_blocks, C, self.block_size, self.block_size
        )

        B_blocked, C_blocked, h, w = x_blocked.shape
        n = h * w
        qkv = self.qkv(x_blocked)
        qkv = qkv.reshape(B_blocked, 3, C_blocked, n)
        qkv = qkv.permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.reshape(B_blocked, self.num_heads, self.head_dim, n)
        k = k.reshape(B_blocked, self.num_heads, self.head_dim, n)
        v = v.reshape(B_blocked, self.num_heads, self.head_dim, n)

        attn = torch.einsum('bhdn, bhdm -> bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhnm, bhdm -> bhdn', attn, v)
        out = out.contiguous().reshape(B_blocked, C_blocked, h, w)
        out = self.proj_drop(self.proj(out))

        out = out.reshape(
            B, num_blocks_h, num_blocks_w, C, self.block_size, self.block_size
        ).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        x = x + local_feat + out
        x = x + self.mlp(x)
        return x


class MobileViTBlock(nn.Module):
    """[official: MobileViTBlock]"""

    def __init__(self, in_channels, out_channels, stride=1, num_heads=4,
                 block_size=8, dropout=0.):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, stride=stride)
        assert out_channels % num_heads == 0, (
            f"out_channels={out_channels} must be an integer multiple of "
            f"num_heads={num_heads}"
        )
        self.attention = BlockAttention(out_channels, num_heads, block_size,
                                        dropout=dropout)
        self.conv2 = ConvLayer(out_channels, out_channels, groups=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        x = self.conv2(x)
        return x


class MobileViT_S_Light(nn.Module):
    """Lightweight MobileViT-S encoder [official: MobileViT_S_Light]."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        self.stem = ConvLayer(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.stage1 = nn.Sequential(
            MobileViTBlock(32, 64, stride=2, num_heads=4, block_size=8),
            MobileViTBlock(64, 64, stride=1, num_heads=4, block_size=8)
        )
        self.stage2 = nn.Sequential(
            MobileViTBlock(64, 128, stride=2, num_heads=8, block_size=8),
            MobileViTBlock(128, 128, stride=1, num_heads=8, block_size=8)
        )
        self.stage3 = nn.Sequential(
            MobileViTBlock(128, 256, stride=2, num_heads=8, block_size=8),
            MobileViTBlock(256, 256, stride=1, num_heads=8, block_size=8)
        )

        self.proj1 = ConvLayer(64, 32, kernel_size=1, padding=0)
        self.proj2 = ConvLayer(128, 32, kernel_size=1, padding=0)
        self.proj3 = ConvLayer(256, 32, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.stem(x)
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)

        feat1 = self.proj1(feat1)
        feat2 = self.proj2(feat2)
        feat3 = self.proj3(feat3)

        return [feat1, feat2, feat3]


class CBAM(nn.Module):
    """Lightweight CBAM [official: CBAM]."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_att


class PositionalEncoding(nn.Module):
    """Positional encoding (d_model=32) [official: PositionalEncoding].

    NOTE: the 64x64 ``pe`` buffer is PERSISTENT (part of the released
    state dicts) and caps the input at 1024x1024 (feat3 = input/16 = 64).
    """

    def __init__(self, d_model=32, H=64, W=64):
        super().__init__()
        self.d_model = d_model
        pos_x = torch.arange(W, dtype=torch.float32).repeat(H, 1)
        pos_y = torch.arange(H, dtype=torch.float32).repeat(W, 1).t()
        pos = torch.stack([pos_x, pos_y], dim=0)
        pe = torch.zeros(1, d_model, H, W)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[0, ::2, :, :] = torch.sin(pos[0:1, :, :] * div_term[None, :, None, None])
        pe[0, 1::2, :, :] = torch.cos(pos[1:2, :, :] * div_term[None, :, None, None])
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :x.size(2), :x.size(3)]


class LightCrossAttention(nn.Module):
    """Lightweight cross-modal attention [official: LightCrossAttention]."""

    def __init__(self, d_model=32, num_heads=4, block_size=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model={d_model} must be an integer multiple of "
            f"num_heads={num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.scale = self.head_dim ** (-0.5)
        self.dropout = nn.Dropout(dropout)
        self.proj_q = nn.Conv2d(d_model, d_model, 1)
        self.proj_k = nn.Conv2d(d_model, d_model, 1)
        self.proj_v = nn.Conv2d(d_model, d_model, 1)
        self.proj_out = nn.Conv2d(d_model, d_model, 1)
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, q, k, v):
        B, C, H, W = q.shape
        q_original = q
        num_blocks_h = H // self.block_size
        num_blocks_w = W // self.block_size
        num_blocks = num_blocks_h * num_blocks_w

        def blockify(x):
            return x.reshape(
                B, C, num_blocks_h, self.block_size, num_blocks_w,
                self.block_size
            ).permute(0, 2, 4, 1, 3, 5).reshape(
                B * num_blocks, C, self.block_size, self.block_size
            )

        q_blocked = blockify(q)
        k_blocked = blockify(k)
        v_blocked = blockify(v)

        B_blocked, C_blocked, h, w = q_blocked.shape
        n = h * w

        q = self.proj_q(q_blocked)
        k = self.proj_k(k_blocked)
        v = self.proj_v(v_blocked)

        q = q.reshape(B_blocked, self.num_heads, self.head_dim, n)
        k = k.reshape(B_blocked, self.num_heads, self.head_dim, n)
        v = v.reshape(B_blocked, self.num_heads, self.head_dim, n)

        attn = torch.einsum('bhdn, bhdm -> bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhnm, bhdm -> bhdn', attn, v)
        out = out.contiguous().reshape(B_blocked, C_blocked, h, w)
        out = self.proj_out(out)

        out = out.reshape(
            B, num_blocks_h, num_blocks_w, C, self.block_size, self.block_size
        ).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return self.norm(out + q_original)


class LightTransformerBlock(nn.Module):
    """Lightweight transformer [official: LightTransformerBlock]."""

    def __init__(self, d_model=32, num_heads=4, hidden_dim=64, block_size=8,
                 dropout=0.1):
        super().__init__()
        self.self_attn = LightCrossAttention(d_model, num_heads, block_size, dropout)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, d_model, 1)
        )
        self.norm = nn.BatchNorm2d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.self_attn(x, x, x)
        x = x + self.dropout(self.ffn(self.norm(x)))
        return x


class HeightPredTransformer(nn.Module):
    """RDAH-Net model [official: HeightPredTransformer].

    forward(depth, img) -> height_pred [B,1,H,W] (nDSM, metres).
    NOTE the ARGUMENT ORDER: depth FIRST, then img (official contract).
    """

    def __init__(self, d_model=32, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 1. lightweight MobileViT-S dual-branch encoders
        self.depth_encoder = MobileViT_S_Light(in_channels=1)
        self.img_encoder = MobileViT_S_Light(in_channels=3)

        # 2. CBAM attention
        self.cbam_blocks = nn.ModuleList([
            CBAM(d_model),
            CBAM(d_model),
            CBAM(d_model)
        ])

        # 3. lightweight cross-modal fusion (bidirectional)
        self.cross_attn_blocks = nn.ModuleList([
            LightCrossAttention(d_model, num_heads, block_size=8),
            LightCrossAttention(d_model, num_heads, block_size=8),
            LightCrossAttention(d_model, num_heads, block_size=8)
        ])
        self.rev_cross_attn_blocks = nn.ModuleList([
            LightCrossAttention(d_model, num_heads, block_size=8),
            LightCrossAttention(d_model, num_heads, block_size=8),
            LightCrossAttention(d_model, num_heads, block_size=8)
        ])

        # 4. lightweight global transformer
        self.pos_encoding = PositionalEncoding(d_model, H=64, W=64)
        self.global_transformer = LightTransformerBlock(d_model, num_heads,
                                                        hidden_dim=64)

        # 5. lightweight decoder (channel-reduced)
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(d_model, 16, 1),
            nn.Conv2d(d_model, 32, 1),
            nn.Conv2d(d_model, 64, 1)
        ])

        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 64 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1)
        )

    def forward(self, depth, img):
        B = depth.shape[0]

        # step 1: multi-scale feature extraction
        depth_feats = self.depth_encoder(depth)
        img_feats = self.img_encoder(img)

        # CBAM enhancement
        depth_feats = [self.cbam_blocks[i](f) for i, f in enumerate(depth_feats)]
        img_feats = [self.cbam_blocks[i](f) for i, f in enumerate(img_feats)]

        # step 2: hierarchical bidirectional fusion
        fused_feats = []
        for i in range(3):
            feat1 = self.cross_attn_blocks[i](
                q=depth_feats[i], k=img_feats[i], v=img_feats[i]
            )
            feat2 = self.rev_cross_attn_blocks[i](
                q=img_feats[i], k=depth_feats[i], v=depth_feats[i]
            )
            fused_feat = (feat1 + feat2) / 2
            fused_feats.append(fused_feat)

        # global transformer
        global_fused_feat = self.pos_encoding(fused_feats[2])
        global_fused_feat = self.global_transformer(global_fused_feat)

        # step 3: decode
        x = global_fused_feat
        x = self.decoder[0:4](x)
        target_size1 = x.shape[2:]
        skip_feat = self.skip_projs[2](fused_feats[2])
        skip_feat = F.interpolate(skip_feat, size=target_size1, mode='bilinear',
                                  align_corners=False)
        x = x + skip_feat

        x = self.decoder[4:8](x)
        target_size2 = x.shape[2:]
        skip_feat = self.skip_projs[1](fused_feats[1])
        skip_feat = F.interpolate(skip_feat, size=target_size2, mode='bilinear',
                                  align_corners=False)
        x = x + skip_feat

        x = self.decoder[8:12](x)
        target_size3 = x.shape[2:]
        skip_feat = self.skip_projs[0](fused_feats[0])
        skip_feat = F.interpolate(skip_feat, size=target_size3, mode='bilinear',
                                  align_corners=False)
        x = x + skip_feat

        height_pred = self.decoder[12:](x)

        return height_pred


def rdah_input_divisibility() -> int:
    """H and W must be divisible by this (BlockAttention 8x8 windows at the
    H/4, H/8 and H/16 encoder scales => lcm constraint 128)."""
    return 128


def rdah_max_input_size() -> int:
    """Max supported H/W for the released checkpoints (PositionalEncoding
    64x64 buffer = input/16)."""
    return 1024
