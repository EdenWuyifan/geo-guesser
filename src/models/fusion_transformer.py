from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FusionOutput:
    fused: torch.Tensor  # (B, D)
    tokens: torch.Tensor  # (B, V, D)


class DirectionalFusionTransformer(nn.Module):
    """
    Learns to fuse 4 view embeddings with direction embeddings + transformer encoder.

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
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # Direction embeddings for N/E/S/W (fixed V=4)
        self.dir_embed = nn.Embedding(4, embed_dim)

        if use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls = None

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
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.dir_embed.weight, std=0.02)
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, view_tokens: torch.Tensor) -> FusionOutput:
        """
        view_tokens: (B, V=4, D)
        """
        b, v, d = view_tokens.shape
        if v != 4:
            raise ValueError(f"Expected V=4 views; got {v}")

        dir_ids = (
            torch.arange(4, device=view_tokens.device).view(1, 4).expand(b, 4)
        )  # (B,4)
        tokens = view_tokens + self.dir_embed(dir_ids)  # (B,4,D)

        if self.use_cls_token:
            cls = self.cls.expand(b, 1, d)  # (B,1,D)
            tokens = torch.cat([cls, tokens], dim=1)  # (B,5,D)

        enc = self.encoder(tokens)  # (B, 5, D) or (B,4,D)
        enc = self.norm(enc)

        if self.use_cls_token:
            fused = enc[:, 0]  # (B,D)
            per_view = enc[:, 1:]  # (B,4,D)
        else:
            fused = enc.mean(dim=1)
            per_view = enc

        return FusionOutput(fused=fused, tokens=per_view)
