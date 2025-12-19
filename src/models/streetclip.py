from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

from .dinov3 import EncoderOutput
from .lora import add_lora_to_attention


class StreetCLIPEncoder(nn.Module):
    """
    Hugging Face StreetCLIP encoder for per-view embeddings.

    Model id (as requested):
      geolocal/StreetCLIP

    Outputs:
      CLIP image embedding from visual_projection(pooler_output) with optional L2 norm.
    """

    def __init__(
        self,
        model_id: str = "geolocal/StreetCLIP",
        freeze: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_layers: int = 4,
        trainable_layers: int = 0,
        normalize: bool = False,
    ) -> None:
        """
        Args:
            model_id: HuggingFace model identifier
            freeze: Whether to freeze backbone (if True, only LoRA params and any unfrozen
                blocks are trainable)
            torch_dtype: Optional dtype for model weights
            use_lora: Enable LoRA adapters on last N vision transformer blocks
            lora_rank: LoRA rank (r)
            lora_alpha: LoRA scaling factor
            lora_layers: Number of last transformer blocks to add LoRA to
            trainable_layers: Unfreeze this many final transformer blocks (in addition to LoRA)
            normalize: If True, L2-normalize embeddings (CLIP-style)
        """
        super().__init__()
        self.model_id = model_id
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_layers = lora_layers
        self.trainable_layers = trainable_layers
        self.normalize = normalize

        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )

        self.embed_dim = int(self.model.visual_projection.out_features)

        self._blocks = self._find_blocks()
        self.trainable_block_indices: list[int] = []

        if freeze:
            self.freeze_backbone()

        if use_lora:
            self._add_lora_adapters()
        if trainable_layers > 0:
            self.unfreeze_last_blocks(trainable_layers)

    @staticmethod
    def processor_params(model_id: str) -> dict:
        """
        Returns recommended preprocessing parameters from HF ImageProcessor.
        Use these in your torchvision transforms for consistency.
        """
        proc = AutoImageProcessor.from_pretrained(model_id)
        size = proc.size
        image_size = size.get("height") or size.get("shortest_edge") or 224
        mean = tuple(proc.image_mean)
        std = tuple(proc.image_std)
        return {"image_size": int(image_size), "mean": mean, "std": std}

    def freeze_backbone(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def _find_blocks(self) -> nn.ModuleList:
        """Locate vision transformer blocks ModuleList."""
        vision = getattr(self.model, "vision_model", None)
        if vision is None:
            raise ValueError("CLIP vision model missing vision_model; check architecture.")

        blocks = None
        if hasattr(vision, "encoder"):
            enc = vision.encoder
            if hasattr(enc, "layers"):
                blocks = enc.layers
            elif hasattr(enc, "layer"):
                blocks = enc.layer
            elif hasattr(enc, "blocks"):
                blocks = enc.blocks

        if blocks is None:
            for name, module in vision.named_modules():
                if isinstance(module, nn.ModuleList) and (
                    "layers" in name.lower() or "blocks" in name.lower()
                ):
                    blocks = module
                    break

        if blocks is None:
            raise ValueError(
                "Could not find transformer blocks in StreetCLIP vision model. "
                "Please check the model architecture."
            )

        return blocks

    def get_blocks(self) -> nn.ModuleList:
        return self._blocks

    def get_trainable_block_indices(self) -> list[int]:
        return getattr(self, "trainable_block_indices", [])

    def unfreeze_last_blocks(self, num_layers: int) -> None:
        if num_layers <= 0:
            self.trainable_block_indices = []
            return

        num_blocks = len(self._blocks)
        start_idx = max(0, num_blocks - num_layers)
        self.trainable_block_indices = list(range(start_idx, num_blocks))
        for i in self.trainable_block_indices:
            for p in self._blocks[i].parameters():
                p.requires_grad = True

    def _add_lora_adapters(self) -> None:
        """Add LoRA adapters to the last N vision transformer blocks."""
        num_blocks = len(self._blocks)
        start_idx = max(0, num_blocks - self.lora_layers)

        for i in range(start_idx, num_blocks):
            block = self._blocks[i]
            attn = None
            if hasattr(block, "self_attn"):
                attn = block.self_attn
            elif hasattr(block, "attn"):
                attn = block.attn
            elif hasattr(block, "attention"):
                attn = block.attention

            if attn is None:
                continue

            try:
                add_lora_to_attention(
                    attn,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    target_modules=["q", "v"],
                )
            except Exception:
                continue

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        x: (B, V, 3, H, W) already normalized/resized.
        """
        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w)

        vision_out = self.model(pixel_values=x)
        embeds = vision_out.image_embeds
        if self.normalize:
            embeds = F.normalize(embeds, p=2, dim=-1)

        embeds = embeds.view(b, v, -1)
        return EncoderOutput(embeddings=embeds)
