from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoImageProcessor

from .lora import add_lora_to_attention


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
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_layers: int = 4,
    ) -> None:
        """
        Args:
            model_id: HuggingFace model identifier
            freeze: Whether to freeze backbone (if True, only LoRA params are trainable)
            torch_dtype: Optional dtype for model weights
            use_lora: Enable LoRA adapters on last N blocks
            lora_rank: LoRA rank (r)
            lora_alpha: LoRA scaling factor
            lora_layers: Number of last transformer blocks to add LoRA to
        """
        super().__init__()
        self.model_id = model_id
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_layers = lora_layers

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

        if use_lora:
            self._add_lora_adapters()

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

    def _add_lora_adapters(self) -> None:
        """Add LoRA adapters to the last N transformer blocks."""
        # Find transformer blocks in the backbone
        # DINOv3 typically has .encoder.blocks or .vit.encoder.layer
        blocks = None
        if hasattr(self.backbone, "encoder") and hasattr(
            self.backbone.encoder, "blocks"
        ):
            blocks = self.backbone.encoder.blocks
        elif hasattr(self.backbone, "vit") and hasattr(self.backbone.vit, "encoder"):
            if hasattr(self.backbone.vit.encoder, "layer"):
                blocks = self.backbone.vit.encoder.layer
            elif hasattr(self.backbone.vit.encoder, "blocks"):
                blocks = self.backbone.vit.encoder.blocks
        elif hasattr(self.backbone, "blocks"):
            blocks = self.backbone.blocks
        else:
            # Try to find blocks by iterating
            for name, module in self.backbone.named_modules():
                if "block" in name.lower() or "layer" in name.lower():
                    if isinstance(module, nn.ModuleList):
                        blocks = module
                        break

        if blocks is None:
            raise ValueError(
                "Could not find transformer blocks in DINOv3 backbone. "
                "Please check the model architecture."
            )

        num_blocks = len(blocks)
        start_idx = max(0, num_blocks - self.lora_layers)

        # Add LoRA to last N blocks
        for i in range(start_idx, num_blocks):
            block = blocks[i]
            # Find attention module
            attn = None
            if hasattr(block, "attn"):
                attn = block.attn
            elif hasattr(block, "attention"):
                attn = block.attention
            elif hasattr(block, "self_attn"):
                attn = block.self_attn

            if attn is not None:
                try:
                    add_lora_to_attention(
                        attn,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        target_modules=["q", "v"],
                    )
                except Exception:
                    # If this block's attention structure doesn't match, skip it
                    continue

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
