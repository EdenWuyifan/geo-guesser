from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .dinov3 import DinoV3Encoder, EncoderOutput
from .fusion_transformer import DirectionalFusionTransformer, FusionOutput


@dataclass
class StateHeadOutput:
    logits: torch.Tensor  # (B, num_states)
    probs: torch.Tensor  # (B, num_states)


class StateHead(nn.Module):
    def __init__(self, embed_dim: int, num_states: int) -> None:
        super().__init__()
        self.num_states = num_states
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_states),
        )

    def forward(self, fused: torch.Tensor) -> StateHeadOutput:
        logits = self.net(fused)
        probs = torch.softmax(logits, dim=-1)
        return StateHeadOutput(logits=logits, probs=probs)


@dataclass
class GeoHeadOutput:
    cell_logits: torch.Tensor  # (B, num_cells)
    cell_probs: torch.Tensor  # (B, num_cells)
    residual: torch.Tensor  # (B, 2)  (Δlat, Δlon) in degrees by default


class GeoHead(nn.Module):
    """
    Predict:
      - geo-cell class distribution
      - residual (Δlat, Δlon) to refine centroid
    """

    def __init__(self, embed_dim: int, num_cells: int) -> None:
        super().__init__()
        self.num_cells = num_cells
        self.cell_net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_cells),
        )
        self.res_net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, fused: torch.Tensor) -> GeoHeadOutput:
        cell_logits = self.cell_net(fused)
        cell_probs = torch.softmax(cell_logits, dim=-1)
        residual = self.res_net(fused)
        return GeoHeadOutput(
            cell_logits=cell_logits, cell_probs=cell_probs, residual=residual
        )


@dataclass
class ModelOutput:
    state: StateHeadOutput
    geo: GeoHeadOutput
    fused: torch.Tensor  # (B, D)


class GeoGuessrModel(nn.Module):
    def __init__(
        self,
        encoder: DinoV3Encoder,
        fusion: DirectionalFusionTransformer,
        state_head: StateHead,
        geo_head: GeoHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion
        self.state_head = state_head
        self.geo_head = geo_head

    def forward(self, images: torch.Tensor) -> ModelOutput:
        """
        images: (B, 4, 3, H, W)
        """
        enc: EncoderOutput = self.encoder(images)
        fus: FusionOutput = self.fusion(enc.embeddings)
        state = self.state_head(fus.fused)
        geo = self.geo_head(fus.fused)
        return ModelOutput(state=state, geo=geo, fused=fus.fused)
