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


@dataclass
class MixtureGeoHeadOutput:
    """Mixture of Gaussians output for multi-modal geolocation."""

    means: torch.Tensor  # (B, K, 2) - K mixture component means [lat, lon]
    covariances: torch.Tensor  # (B, K, 2, 2) - covariance matrices
    weights: torch.Tensor  # (B, K) - mixture weights (softmax normalized)
    # Optional: keep cell logits for auxiliary loss
    cell_logits: torch.Tensor | None = None  # (B, num_cells) if use_cell_aux=True


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


class MixtureGeoHead(nn.Module):
    """
    Mixture-of-Experts regression head for multi-modal geolocation.

    Predicts K Gaussian components, each with:
    - Mean (lat, lon)
    - Covariance matrix (2x2, parameterized as lower triangular)
    - Mixture weight

    Handles ambiguous scenes (e.g., suburban roads) better than single-point regression.
    """

    def __init__(
        self,
        embed_dim: int,
        num_components: int = 5,
        use_cell_aux: bool = False,
        num_cells: int | None = None,
        min_cov: float = 1e-3,
    ) -> None:
        """
        Args:
            embed_dim: Input embedding dimension
            num_components: Number of mixture components (K)
            use_cell_aux: If True, also predict cell logits for auxiliary loss
            num_cells: Number of geo-cells (required if use_cell_aux=True)
            min_cov: Minimum covariance value for numerical stability
        """
        super().__init__()
        self.num_components = num_components
        self.use_cell_aux = use_cell_aux
        self.min_cov = min_cov

        # Each component needs:
        # - Mean: 2 (lat, lon)
        # - Covariance (lower triangular): 3 (L11, L21, L22) for 2x2 pos-def matrix
        # - Weight: 1 (will be softmaxed)
        # Total: K * (2 + 3 + 1) = K * 6 parameters
        self.mixture_net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_components * 6),
        )

        if use_cell_aux:
            if num_cells is None:
                raise ValueError("num_cells required when use_cell_aux=True")
            self.cell_net = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_cells),
            )
        else:
            self.cell_net = None

    def forward(self, fused: torch.Tensor) -> MixtureGeoHeadOutput:
        """
        Args:
            fused: (B, embed_dim) fused representation

        Returns:
            MixtureGeoHeadOutput with means, covariances, weights, and optional cell_logits
        """
        B = fused.shape[0]
        K = self.num_components

        # Predict mixture parameters: (B, K*6)
        params = self.mixture_net(fused)  # (B, K*6)
        params = params.view(B, K, 6)  # (B, K, 6)

        # Split into means, covariance params, and weights
        means = params[:, :, :2]  # (B, K, 2) [lat, lon]
        cov_params = params[:, :, 2:5]  # (B, K, 3) [L11, L21, L22]
        log_weights = params[:, :, 5]  # (B, K)

        # Normalize mixture weights
        weights = torch.softmax(log_weights, dim=-1)  # (B, K)

        # Build positive-definite covariance matrices from lower triangular
        # L = [[L11, 0], [L21, L22]] -> Cov = L @ L.T
        # Use softplus to ensure positive diagonal elements
        L11 = torch.nn.functional.softplus(cov_params[:, :, 0]) + self.min_cov  # (B, K)
        L21 = cov_params[:, :, 1]  # (B, K)
        L22 = torch.nn.functional.softplus(cov_params[:, :, 2]) + self.min_cov  # (B, K)

        # Construct lower triangular matrices
        L = torch.zeros(B, K, 2, 2, device=fused.device, dtype=fused.dtype)
        L[:, :, 0, 0] = L11
        L[:, :, 1, 0] = L21
        L[:, :, 1, 1] = L22

        # Compute covariance: Cov = L @ L.T
        covariances = L @ L.transpose(-2, -1)  # (B, K, 2, 2)

        # Add regularization to diagonal to ensure positive definiteness
        # This helps with numerical stability - use larger value for safety
        identity = (
            torch.eye(2, device=fused.device, dtype=fused.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, 2, 2)
        covariances = covariances + identity * (self.min_cov * 10)  # (B, K, 2, 2)

        # Optional: cell classification for auxiliary loss
        cell_logits = None
        if self.use_cell_aux and self.cell_net is not None:
            cell_logits = self.cell_net(fused)  # (B, num_cells)

        return MixtureGeoHeadOutput(
            means=means,
            covariances=covariances,
            weights=weights,
            cell_logits=cell_logits,
        )


@dataclass
class ModelOutput:
    state: StateHeadOutput
    geo: GeoHeadOutput | MixtureGeoHeadOutput
    fused: torch.Tensor  # (B, D)
    view_tokens: torch.Tensor | None = None  # (B, V=4, D) for self-supervised losses


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
        return ModelOutput(
            state=state, geo=geo, fused=fus.fused, view_tokens=enc.embeddings
        )
