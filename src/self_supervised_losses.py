from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import haversine_km


class ViewConsistencyLoss(nn.Module):
    """
    Contrastive loss for view consistency: embeddings of 4 views from the same sample
    should be close; views from different samples should be far.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        """
        Args:
            temperature: Temperature parameter for contrastive loss (lower = sharper)
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self, view_tokens: torch.Tensor, sample_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            view_tokens: (B, V=4, D) embeddings for each view
            sample_ids: (B,) optional sample IDs for hard negative mining (unused for now)

        Returns:
            Contrastive loss scalar
        """
        B, V, D = view_tokens.shape
        device = view_tokens.device

        # Normalize embeddings
        view_tokens = F.normalize(view_tokens, p=2, dim=-1)  # (B, V, D)

        # Flatten to (B*V, D) for easier computation
        views_flat = view_tokens.view(B * V, D)  # (B*V, D)

        # Compute similarity matrix: (B*V, B*V)
        similarity = (
            torch.matmul(views_flat, views_flat.T) / self.temperature
        )  # (B*V, B*V)

        # Create positive mask: same sample (but different views)
        pos_mask = torch.zeros(B * V, B * V, device=device, dtype=torch.bool)
        for i in range(B):
            start_idx = i * V
            end_idx = (i + 1) * V
            # All views from same sample are positives (excluding self)
            pos_mask[start_idx:end_idx, start_idx:end_idx] = True
            pos_mask[start_idx:end_idx, start_idx:end_idx].fill_diagonal_(False)

        # InfoNCE loss: for each view, maximize similarity to positives
        loss = 0.0
        for i in range(B * V):
            pos_indices = pos_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if len(pos_indices) == 0:
                continue

            # Log-softmax over all similarities
            log_probs = F.log_softmax(similarity[i], dim=0)  # (B*V,)

            # Sum log-probabilities of positive pairs
            pos_log_probs = log_probs[pos_indices]
            loss -= pos_log_probs.mean()

        return loss / (B * V) if B * V > 0 else torch.tensor(0.0, device=device)


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
    ) -> None:
        """
        Args:
            positive_threshold_km: Distance threshold for positive pairs (km)
            negative_threshold_km: Distance threshold for negative pairs (km)
            temperature: Temperature parameter for contrastive loss
            soft_weighting: If True, use soft weights based on distance
        """
        super().__init__()
        self.positive_threshold_km = positive_threshold_km
        self.negative_threshold_km = negative_threshold_km
        self.temperature = temperature
        self.soft_weighting = soft_weighting

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, D)
        latlon: torch.Tensor,  # (B, 2) [lat, lon] in degrees
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) fused embeddings
            latlon: (B, 2) ground truth lat/lon coordinates

        Returns:
            Contrastive loss scalar
        """
        B, D = embeddings.shape
        device = embeddings.device

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

        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)

        # Create soft weights based on distance
        if self.soft_weighting:
            # Positive weight: higher for closer pairs, decays with distance
            pos_weights = torch.exp(
                -distances_km / self.positive_threshold_km
            )  # (B, B)
            pos_weights.fill_diagonal_(0)  # Exclude self

            # Negative weight: higher for farther pairs
            neg_weights = torch.clamp(
                distances_km / self.negative_threshold_km, min=0.0, max=1.0
            )  # (B, B)
            neg_weights.fill_diagonal_(0)  # Exclude self
        else:
            # Hard thresholding
            pos_mask = distances_km < self.positive_threshold_km
            neg_mask = distances_km > self.negative_threshold_km
            pos_weights = pos_mask.float()
            neg_weights = neg_mask.float()
            pos_weights.fill_diagonal_(0)
            neg_weights.fill_diagonal_(0)

        # Contrastive loss: maximize similarity for positives, minimize for negatives
        # Use InfoNCE-style loss: -log(exp(pos) / (exp(pos) + exp(neg)))
        loss = 0.0
        for i in range(B):
            # Positive pairs (weighted)
            pos_weight = pos_weights[i]  # (B,)
            pos_mask = pos_weight > 1e-6
            if pos_mask.sum() > 0:
                # Weighted positive similarities
                pos_sim = similarity[i] * pos_weight  # (B,)
                pos_logsumexp = torch.logsumexp(pos_sim[pos_mask], dim=0)
            else:
                pos_logsumexp = torch.tensor(0.0, device=device)

            # All other pairs (including negatives)
            all_sim = similarity[i]  # (B,)
            all_sim = all_sim.clone()
            all_sim[i] = float("-inf")  # Exclude self
            all_logsumexp = torch.logsumexp(all_sim, dim=0)

            # InfoNCE: -log(exp(pos) / exp(all)) = -pos + all
            loss += -pos_logsumexp + all_logsumexp

        return loss / B if B > 0 else torch.tensor(0.0, device=device)


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
    ) -> dict[str, torch.Tensor]:
        """
        Compute self-supervised losses.

        Returns:
            Dictionary with loss components
        """
        losses = {}

        # View consistency loss
        if self.lambda_view_consistency > 0:
            view_loss = self.view_consistency_loss(view_tokens)
            losses["view_consistency"] = view_loss

        # Geo-distance contrastive loss
        if self.lambda_geo_distance > 0:
            geo_loss = self.geo_distance_loss(fused_embeddings, latlon)
            losses["geo_distance"] = geo_loss

        # Total self-supervised loss
        total = self.lambda_view_consistency * losses.get(
            "view_consistency", 0.0
        ) + self.lambda_geo_distance * losses.get("geo_distance", 0.0)
        losses["total"] = total

        return losses
