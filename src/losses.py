from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in [0,1).")
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C)
        target: (B,)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        # Negative log-likelihood for true class
        nll = F.nll_loss(log_probs, target, reduction="none")
        # Uniform smoothing
        smooth = -log_probs.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth).mean()


def haversine_km(latlon1: torch.Tensor, latlon2: torch.Tensor) -> torch.Tensor:
    """
    latlon1, latlon2: (B,2) in degrees [lat, lon]
    returns: (B,) distances in km
    """
    lat1 = torch.deg2rad(latlon1[:, 0])
    lon1 = torch.deg2rad(latlon1[:, 1])
    lat2 = torch.deg2rad(latlon2[:, 0])
    lon2 = torch.deg2rad(latlon2[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.arcsin(torch.clamp(torch.sqrt(a), 0.0, 1.0))
    R = 6371.0
    return R * c


class HaversineLoss(nn.Module):
    """
    A simple regression loss based on Haversine distance (km).
    Often combined with cell classification.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none")
        self.reduction = reduction

    def forward(
        self, pred_latlon: torch.Tensor, true_latlon: torch.Tensor
    ) -> torch.Tensor:
        d_km = haversine_km(pred_latlon, true_latlon)
        d = d_km / 5000.0
        if self.reduction == "mean":
            return d.mean()
        if self.reduction == "sum":
            return d.sum()
        return d


@dataclass
class LossOutput:
    total: torch.Tensor
    parts: Dict[str, torch.Tensor]


class MixtureNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for mixture of Gaussians.
    Handles multi-modal geolocation uncertainty.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none")
        self.reduction = reduction

    def forward(
        self,
        means: torch.Tensor,  # (B, K, 2)
        covariances: torch.Tensor,  # (B, K, 2, 2)
        weights: torch.Tensor,  # (B, K)
        true_latlon: torch.Tensor,  # (B, 2)
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood of true_latlon under mixture distribution.
        """
        B, K, _ = means.shape

        # Convert to float32 for MultivariateNormal (doesn't support bfloat16/half)
        means_f32 = means.float()
        covariances_f32 = covariances.float()
        true_latlon_f32 = true_latlon.float()
        weights_f32 = weights.float()

        # Ensure positive definiteness: add regularization to diagonal
        # This is a safety measure in case some matrices are still problematic
        min_eigenvalue = 1e-4
        identity = torch.eye(
            2, device=covariances_f32.device, dtype=covariances_f32.dtype
        ).unsqueeze(
            0
        )  # (1, 2, 2)
        # Add regularization to all covariance matrices at once
        covariances_f32 = (
            covariances_f32 + identity.unsqueeze(0) * min_eigenvalue
        )  # (B, K, 2, 2)

        # Compute log-probability for each mixture component
        log_probs = []
        for k in range(K):
            try:
                comp_dist = MultivariateNormal(
                    loc=means_f32[:, k, :],  # (B, 2)
                    covariance_matrix=covariances_f32[:, k, :, :],  # (B, 2, 2)
                )
                log_prob_k = comp_dist.log_prob(true_latlon_f32)  # (B,)
                log_probs.append(log_prob_k)
            except Exception:
                # Fallback: use diagonal approximation if full covariance fails
                cov_k = covariances_f32[:, k, :, :]  # (B, 2, 2)
                # Extract diagonal and ensure positive
                diag = torch.diagonal(cov_k, dim1=-2, dim2=-1)  # (B, 2)
                diag = torch.clamp(diag, min=min_eigenvalue)
                # Use independent Gaussians as fallback
                from torch.distributions import Normal

                dist_lat = Normal(means_f32[:, k, 0], torch.sqrt(diag[:, 0]))
                dist_lon = Normal(means_f32[:, k, 1], torch.sqrt(diag[:, 1]))
                log_prob_k = dist_lat.log_prob(
                    true_latlon_f32[:, 0]
                ) + dist_lon.log_prob(true_latlon_f32[:, 1])
                log_probs.append(log_prob_k)

        # Stack: (B, K)
        log_probs_stack = torch.stack(log_probs, dim=1)
        log_weights = torch.log(weights_f32 + 1e-8)  # (B, K)

        # Log-sum-exp for numerical stability: log(sum_k w_k * exp(log_prob_k))
        # = log(sum_k exp(log_weights_k + log_prob_k))
        log_mixture_prob = torch.logsumexp(log_weights + log_probs_stack, dim=1)  # (B,)
        nll = -log_mixture_prob  # (B,)

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        state_smoothing: float = 0.1,
        lambda_state: float = 1.0,
        lambda_cell: float = 1.0,
        lambda_gps: float = 1.0,
        gps_loss_type: str = "haversine",
    ) -> None:
        super().__init__()
        self.state_loss = LabelSmoothingCrossEntropy(state_smoothing)
        self.cell_loss = nn.CrossEntropyLoss()

        if gps_loss_type == "haversine":
            self.gps_loss = HaversineLoss()
        elif gps_loss_type == "huber":
            self.gps_loss = nn.SmoothL1Loss()
        else:
            raise ValueError("gps_loss_type must be 'haversine' or 'huber'")

        self.lambda_state = lambda_state
        self.lambda_cell = lambda_cell
        self.lambda_gps = lambda_gps

    def forward(
        self,
        state_logits: torch.Tensor,
        cell_logits: torch.Tensor,
        pred_latlon: torch.Tensor,
        true_state: torch.Tensor,
        true_cell: torch.Tensor,
        true_latlon: torch.Tensor,
    ) -> LossOutput:
        ls = self.state_loss(state_logits, true_state)
        lc = self.cell_loss(cell_logits, true_cell)
        lg = self.gps_loss(pred_latlon, true_latlon)

        total = self.lambda_state * ls + self.lambda_cell * lc + self.lambda_gps * lg
        return LossOutput(total=total, parts={"state": ls, "cell": lc, "gps": lg})


class MixtureMultiTaskLoss(nn.Module):
    """
    Multi-task loss for mixture-of-experts geolocation model.
    Uses NLL for mixture GPS prediction instead of point regression.
    """

    def __init__(
        self,
        state_smoothing: float = 0.1,
        lambda_state: float = 1.0,
        lambda_cell: float = 0.0,  # Optional auxiliary loss
        lambda_gps: float = 1.0,
        use_cell_aux: bool = False,
    ) -> None:
        super().__init__()
        self.state_loss = LabelSmoothingCrossEntropy(state_smoothing)
        self.mixture_gps_loss = MixtureNLLLoss()
        self.use_cell_aux = use_cell_aux

        if use_cell_aux:
            self.cell_loss = nn.CrossEntropyLoss()
        else:
            self.cell_loss = None

        self.lambda_state = lambda_state
        self.lambda_cell = lambda_cell if use_cell_aux else 0.0
        self.lambda_gps = lambda_gps

    def forward(
        self,
        state_logits: torch.Tensor,
        means: torch.Tensor,  # (B, K, 2)
        covariances: torch.Tensor,  # (B, K, 2, 2)
        weights: torch.Tensor,  # (B, K)
        true_state: torch.Tensor,
        true_latlon: torch.Tensor,
        cell_logits: torch.Tensor | None = None,
        true_cell: torch.Tensor | None = None,
    ) -> LossOutput:
        ls = self.state_loss(state_logits, true_state)
        lg = self.mixture_gps_loss(means, covariances, weights, true_latlon)

        parts = {"state": ls, "gps": lg}

        if self.use_cell_aux and cell_logits is not None and true_cell is not None:
            lc = self.cell_loss(cell_logits, true_cell)
            parts["cell"] = lc
            total = (
                self.lambda_state * ls + self.lambda_cell * lc + self.lambda_gps * lg
            )
        else:
            total = self.lambda_state * ls + self.lambda_gps * lg

        return LossOutput(total=total, parts=parts)
