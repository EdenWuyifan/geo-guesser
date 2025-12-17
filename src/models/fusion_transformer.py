from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionOutput:
    fused: torch.Tensor  # (B, D)
    tokens: torch.Tensor  # (B, V, D)


class DirectionalFusionTransformer(nn.Module):
    """
    Improved multi-view fusion with cross-attention, learned positional embeddings,
    and view dropout support.

    Features:
    - Cross-attention between view tokens and global token
    - Learned directional positional embeddings (N/E/S/W)
    - View dropout for training robustness
    - Transformer encoder for view interaction

    Input:
      tokens: (B, V=4, D)
    Output:
      fused: (B, D)
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_global_token: bool = True,
        view_dropout_prob: float = 0.0,
    ) -> None:
        """
        Args:
            embed_dim: Embedding dimension
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_global_token: Whether to use a learnable global token
            view_dropout_prob: Probability of dropping a view during training (0.0 = disabled)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_global_token = use_global_token
        self.view_dropout_prob = view_dropout_prob

        # Learned directional positional embeddings for N/E/S/W
        # More expressive than simple embeddings
        self.dir_pos_embed = nn.Parameter(torch.zeros(4, embed_dim))
        # Optional: learnable direction-specific scaling
        self.dir_scale = nn.Parameter(torch.ones(4))

        # Global token (acts as query for cross-attention)
        if use_global_token:
            self.global_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter("global_token", None)

        # Transformer encoder for view interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention layer: global token attends to view tokens
        if use_global_token:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_attn_norm = nn.LayerNorm(embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Initialize parameters
        nn.init.trunc_normal_(self.dir_pos_embed, std=0.02)
        nn.init.normal_(self.dir_scale, mean=1.0, std=0.02)
        if self.global_token is not None:
            nn.init.trunc_normal_(self.global_token, std=0.02)

    def _apply_view_dropout(
        self, view_tokens: torch.Tensor, training: bool
    ) -> torch.Tensor:
        """
        Randomly drop 1-2 views during training for robustness.
        Replaces dropped views with zero padding.

        Args:
            view_tokens: (B, V=4, D)
            training: Whether in training mode

        Returns:
            view_tokens with some views potentially zeroed out
        """
        if not training or self.view_dropout_prob == 0.0:
            return view_tokens

        b, v, d = view_tokens.shape
        device = view_tokens.device

        # Create mask: each sample independently drops 1-2 views
        mask = torch.ones(b, v, device=device, dtype=torch.bool)

        for i in range(b):
            if torch.rand(1, device=device).item() < self.view_dropout_prob:
                # Drop 1-2 views randomly
                num_drop = torch.randint(1, 3, (1,), device=device).item()
                drop_indices = torch.randperm(v, device=device)[:num_drop]
                mask[i, drop_indices] = False

        # Zero out dropped views
        mask_expanded = mask.unsqueeze(-1)  # (B, V, 1)
        view_tokens = view_tokens * mask_expanded

        return view_tokens

    def forward(
        self, view_tokens: torch.Tensor, training: bool | None = None
    ) -> FusionOutput:
        """
        Args:
            view_tokens: (B, V=4, D) view embeddings
            training: Optional override for training mode (defaults to self.training)

        Returns:
            FusionOutput with fused representation and per-view tokens
        """
        if training is None:
            training = self.training

        b, v, d = view_tokens.shape
        if v != 4:
            raise ValueError(f"Expected V=4 views; got {v}")

        # Apply view dropout during training
        view_tokens = self._apply_view_dropout(view_tokens, training)

        # Add learned directional positional embeddings with scaling
        # More expressive than simple addition
        dir_embeds = self.dir_pos_embed.unsqueeze(0) * self.dir_scale.unsqueeze(
            0
        ).unsqueeze(
            -1
        )  # (1, 4, D)
        tokens = view_tokens + dir_embeds  # (B, 4, D)

        # Pass through transformer encoder for view interaction
        enc_tokens = self.encoder(tokens)  # (B, 4, D)
        enc_tokens = self.norm(enc_tokens)

        # Cross-attention: global token attends to view tokens
        if self.use_global_token:
            global_tok = self.global_token.expand(b, 1, d)  # (B, 1, D)

            # Cross-attention: query=global_token, key/value=view_tokens
            attn_out, _ = self.cross_attn(
                query=global_tok,
                key=enc_tokens,
                value=enc_tokens,
            )  # (B, 1, D)
            fused = self.cross_attn_norm(attn_out.squeeze(1))  # (B, D)
        else:
            # Fallback: weighted mean of view tokens
            # Use attention weights as importance scores
            fused = enc_tokens.mean(dim=1)  # (B, D)

        # Final projection for better representation
        fused = self.output_proj(fused)  # (B, D)

        return FusionOutput(fused=fused, tokens=enc_tokens)
