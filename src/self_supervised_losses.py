from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import haversine_km


def _masked_logsumexp(
    x: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    Compute logsumexp over `x` along `dim`, masking out elements where mask is False.

    mask: broadcastable to x, boolean.
    """
    neg_inf = torch.finfo(x.dtype).min
    x_masked = x.masked_fill(~mask, neg_inf)
    return torch.logsumexp(x_masked, dim=dim)


class ViewConsistencyLoss(nn.Module):
    """
    Contrastive loss for view consistency: embeddings of 4 views from the same sample
    should be close; views from different samples should be far.
    """

    def __init__(self, temperature: float = 0.07, hard_negative_weight: float = 2.0) -> None:
        """
        Args:
            temperature: Temperature parameter for contrastive loss (lower = sharper)
            hard_negative_weight: Multiplier for negatives from the same state (if provided)
        """
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        view_tokens: torch.Tensor,
        state_class: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            view_tokens: (B, V=4, D) embeddings for each view
            state_class: (B,) optional state class IDs to upweight hard negatives

        Returns:
            Contrastive loss scalar
        """
        B, V, D = view_tokens.shape

        # Normalize embeddings
        view_tokens = F.normalize(view_tokens, p=2, dim=-1)  # (B, V, D)

        # Flatten to (N, D) where N=B*V
        views_flat = view_tokens.reshape(B * V, D)  # (B*V, D)
        sample_idx = torch.arange(B, device=view_tokens.device).repeat_interleave(V)  # (N,)

        logits = torch.matmul(views_flat, views_flat.T) / self.temperature  # (N, N)

        # Positives: same sample, different view
        pos_mask = sample_idx[:, None] == sample_idx[None, :]
        pos_mask.fill_diagonal_(False)

        # Valid comparisons: exclude self
        valid_mask = torch.ones_like(pos_mask, dtype=torch.bool)
        valid_mask.fill_diagonal_(False)

        # Hard-negative weighting for negatives from same state (different sample)
        if state_class is not None:
            state_per_view = state_class.to(view_tokens.device)[sample_idx]  # (N,)
            same_state = state_per_view[:, None] == state_per_view[None, :]
            same_sample = sample_idx[:, None] == sample_idx[None, :]
            hard_neg_mask = same_state & (~same_sample)
            neg_weights = torch.ones_like(logits)
            neg_weights = neg_weights + hard_neg_mask.to(logits.dtype) * (
                float(self.hard_negative_weight) - 1.0
            )
        else:
            neg_weights = torch.ones_like(logits)

        # Denominator weights apply to all non-self comparisons
        denom_logits = logits + torch.log(neg_weights + 1e-8)
        denom = _masked_logsumexp(denom_logits, valid_mask, dim=1)  # (N,)

        # Multi-positive numerator: logsumexp over positives
        has_pos = pos_mask.any(dim=1)
        numerator = _masked_logsumexp(logits, pos_mask, dim=1)  # (N,)
        loss = -(numerator - denom)
        loss = loss.masked_fill(~has_pos, 0.0)

        denom_count = has_pos.sum().clamp(min=1).to(loss.dtype)
        return loss.sum() / denom_count


class GeoDistanceContrastiveLoss(nn.Module):
    """
    Contrastive loss based on geographic distance.
    Positives are nearby (within X km), negatives are far, with soft weighting.
    """

    def __init__(
        self,
        positive_threshold_km: float = 100.0,
        negative_threshold_km: float = 1000.0,
        temperature: float = 0.07,
        soft_weighting: bool = True,
        hard_negative_weight: float = 2.0,
    ) -> None:
        """
        Args:
            positive_threshold_km: Distance threshold for positive pairs (km)
            negative_threshold_km: Distance threshold for negative pairs (km)
            temperature: Temperature parameter for contrastive loss
            soft_weighting: If True, use soft weights based on distance
            hard_negative_weight: Multiplier for negatives from the same state (if provided)
        """
        super().__init__()
        self.positive_threshold_km = positive_threshold_km
        self.negative_threshold_km = negative_threshold_km
        self.temperature = temperature
        self.soft_weighting = soft_weighting
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, D)
        latlon: torch.Tensor,  # (B, 2) [lat, lon] in degrees
        state_class: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) fused embeddings
            latlon: (B, 2) ground truth lat/lon coordinates

        Returns:
            Contrastive loss scalar
        """
        B, D = embeddings.shape

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)  # (B, D)

        # Compute pairwise distances (km) - vectorized
        # Expand for broadcasting: (B, 1, 2) and (1, B, 2)
        latlon_i = latlon.unsqueeze(1)  # (B, 1, 2)
        latlon_j = latlon.unsqueeze(0)  # (1, B, 2)

        # Compute distances for all pairs at once
        distances_km = haversine_km(
            latlon_i.expand(B, B, 2).reshape(B * B, 2),
            latlon_j.expand(B, B, 2).reshape(B * B, 2),
        ).view(
            B, B
        )  # (B, B)

        logits = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)
        valid_mask = torch.ones_like(logits, dtype=torch.bool)
        valid_mask.fill_diagonal_(False)

        # Positive weights based on distance (soft) or threshold (hard)
        if self.soft_weighting:
            pos_weights = torch.exp(-distances_km / self.positive_threshold_km)  # (B, B)
        else:
            pos_weights = (distances_km < self.positive_threshold_km).to(logits.dtype)
        pos_weights.fill_diagonal_(0.0)

        # Denominator weighting: upweight hard negatives from same state (different sample)
        if state_class is not None:
            state_class = state_class.to(embeddings.device)
            same_state = state_class[:, None] == state_class[None, :]
            hard_neg_weights = torch.ones_like(logits)
            hard_neg_weights = hard_neg_weights + same_state.to(logits.dtype) * (
                float(self.hard_negative_weight) - 1.0
            )
        else:
            hard_neg_weights = torch.ones_like(logits)
        hard_neg_weights.fill_diagonal_(0.0)

        # Numerator: weighted multi-positive logsumexp
        pos_mask = pos_weights > 1e-6
        has_pos = pos_mask.any(dim=1)
        numerator_logits = logits + torch.log(pos_weights + 1e-8)
        numerator = _masked_logsumexp(numerator_logits, pos_mask, dim=1)  # (B,)

        # Denominator: all non-self comparisons (with hard negative reweighting)
        denom_logits = logits + torch.log(hard_neg_weights + 1e-8)
        denom = _masked_logsumexp(denom_logits, valid_mask, dim=1)  # (B,)

        loss = -(numerator - denom)
        loss = loss.masked_fill(~has_pos, 0.0)
        denom_count = has_pos.sum().clamp(min=1).to(loss.dtype)
        return loss.sum() / denom_count


class HardNegativeSampler:
    """
    Sample hard negatives by geography: same state or adjacent states.
    """

    def __init__(
        self,
        state_classes: torch.Tensor,  # (N,) state class IDs for all samples
        latlon: torch.Tensor,  # (N, 2) lat/lon for all samples
        same_state_weight: float = 0.5,
        adjacent_state_weight: float = 0.3,
    ) -> None:
        """
        Args:
            state_classes: State class IDs for all training samples
            latlon: Lat/lon coordinates for all training samples
            same_state_weight: Probability of sampling from same state
            adjacent_state_weight: Probability of sampling from adjacent states
        """
        self.state_classes = state_classes
        self.latlon = latlon
        self.same_state_weight = same_state_weight
        self.adjacent_state_weight = adjacent_state_weight

        # Build state-to-indices mapping
        self.state_to_indices = {}
        unique_states = torch.unique(state_classes)
        for state in unique_states:
            self.state_to_indices[state.item()] = torch.where(state_classes == state)[0]

        # TODO: Build adjacency graph if state adjacency info is available
        # For now, we'll use geographic proximity as a proxy

    def sample_hard_negatives(
        self, batch_indices: torch.Tensor, num_negatives: int = 1
    ) -> torch.Tensor:
        """
        Sample hard negative indices for given batch indices.

        Args:
            batch_indices: (B,) indices of current batch samples
            num_negatives: Number of negatives to sample per sample

        Returns:
            (B * num_negatives,) indices of hard negative samples
        """
        device = batch_indices.device
        negative_indices = []

        batch_states = self.state_classes[batch_indices.cpu()]

        for i, idx in enumerate(batch_indices.cpu()):
            state = batch_states[i].item()
            same_state_indices = self.state_to_indices[state]

            # Remove current sample from candidates
            candidates = same_state_indices[same_state_indices != idx]

            if len(candidates) > 0 and torch.rand(1).item() < self.same_state_weight:
                # Sample from same state
                neg_idx = candidates[torch.randint(0, len(candidates), (1,))]
            else:
                # Sample from nearby geographic region (same state or nearby)
                # Use all other samples as candidates
                all_indices = torch.arange(len(self.state_classes))
                candidates = all_indices[all_indices != idx]
                neg_idx = candidates[torch.randint(0, len(candidates), (1,))]

            negative_indices.append(neg_idx)

        return torch.stack(negative_indices).to(device)


class SelfSupervisedLoss(nn.Module):
    """
    Combined self-supervised loss for geolocation.
    Includes view consistency and geo-distance contrastive losses.
    """

    def __init__(
        self,
        lambda_view_consistency: float = 0.1,
        lambda_geo_distance: float = 0.1,
        view_temperature: float = 0.07,
        geo_temperature: float = 0.07,
        positive_threshold_km: float = 100.0,
        negative_threshold_km: float = 1000.0,
    ) -> None:
        """
        Args:
            lambda_view_consistency: Weight for view consistency loss
            lambda_geo_distance: Weight for geo-distance contrastive loss
            view_temperature: Temperature for view consistency loss
            geo_temperature: Temperature for geo-distance loss
            positive_threshold_km: Distance threshold for positive pairs (km)
            negative_threshold_km: Distance threshold for negative pairs (km)
        """
        super().__init__()
        self.lambda_view_consistency = lambda_view_consistency
        self.lambda_geo_distance = lambda_geo_distance

        self.view_consistency_loss = ViewConsistencyLoss(temperature=view_temperature)
        self.geo_distance_loss = GeoDistanceContrastiveLoss(
            positive_threshold_km=positive_threshold_km,
            negative_threshold_km=negative_threshold_km,
            temperature=geo_temperature,
            soft_weighting=True,
        )

    def forward(
        self,
        view_tokens: torch.Tensor,  # (B, V=4, D)
        fused_embeddings: torch.Tensor,  # (B, D)
        latlon: torch.Tensor,  # (B, 2)
        state_class: torch.Tensor | None = None,  # (B,)
    ) -> dict[str, torch.Tensor]:
        """
        Compute self-supervised losses.

        Returns:
            Dictionary with loss components
        """
        losses = {}

        # View consistency loss
        if self.lambda_view_consistency > 0:
            view_loss = self.view_consistency_loss(view_tokens, state_class=state_class)
            losses["view_consistency"] = view_loss

        # Geo-distance contrastive loss
        if self.lambda_geo_distance > 0:
            geo_loss = self.geo_distance_loss(
                fused_embeddings, latlon, state_class=state_class
            )
            losses["geo_distance"] = geo_loss

        # Total self-supervised loss
        total = self.lambda_view_consistency * losses.get(
            "view_consistency", 0.0
        ) + self.lambda_geo_distance * losses.get("geo_distance", 0.0)
        losses["total"] = total

        return losses
