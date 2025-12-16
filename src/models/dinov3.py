from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoImageProcessor


@dataclass
class EncoderOutput:
    embeddings: torch.Tensor  # (B, V, D)


class DinoV3Encoder(nn.Module):
    """
    Hugging Face DINOv3 encoder for per-view embeddings.

    Model id (as requested):
      facebook/dinov3-vit7b16-pretrain-lvd1689m

    Outputs:
      CLS token embedding from last_hidden_state[:, 0, :]
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        freeze: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id

        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

        # hidden size inferred from config
        self.embed_dim = getattr(self.config, "hidden_size", None)
        if self.embed_dim is None:
            # Some ViT configs use "dim"
            self.embed_dim = getattr(self.config, "dim", None)
        if self.embed_dim is None:
            raise ValueError("Could not infer embed_dim from model config.")

        if freeze:
            self.freeze_backbone()

    @staticmethod
    def processor_params(model_id: str) -> dict:
        """
        Returns recommended preprocessing parameters from HF ImageProcessor.
        Use these in your torchvision transforms for consistency.
        """
        proc = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        size = proc.size
        # Common patterns: {"height": 224, "width": 224} or {"shortest_edge": 224}
        image_size = size.get("height") or size.get("shortest_edge") or 224
        mean = tuple(proc.image_mean)
        std = tuple(proc.image_std)
        return {"image_size": int(image_size), "mean": mean, "std": std}

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        x: (B, V, 3, H, W) already normalized/resized.
        """
        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w)

        out = self.backbone(pixel_values=x)
        if not hasattr(out, "last_hidden_state"):
            raise RuntimeError(
                "Backbone output missing last_hidden_state; check HF model outputs."
            )

        # CLS token embedding
        cls = out.last_hidden_state[:, 0, :]  # (B*V, D)
        cls = cls.view(b, v, -1)  # (B, V, D)
        return EncoderOutput(embeddings=cls)
