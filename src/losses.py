from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        n_classes = logits.size(-1)
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
        d = haversine_km(pred_latlon, true_latlon)
        if self.reduction == "mean":
            return d.mean()
        if self.reduction == "sum":
            return d.sum()
        return d


@dataclass
class LossOutput:
    total: torch.Tensor
    parts: Dict[str, torch.Tensor]


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
