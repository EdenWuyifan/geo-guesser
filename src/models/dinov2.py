from __future__ import annotations

from typing import Optional

import torch

from .dinov3 import DinoV3Encoder


class DinoV2Encoder(DinoV3Encoder):
    """
    DINOv2 encoder using the same interface as DinoV3Encoder.

    Notes:
      - DINOv2 models are standard Hugging Face models and generally do not require
        `trust_remote_code=True`.
      - Output is the CLS embedding from `last_hidden_state[:, 0, :]`.
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov2-vitb14",
        freeze: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_layers: int = 4,
        trainable_layers: int = 0,
    ) -> None:
        super().__init__(
            model_id=model_id,
            freeze=freeze,
            torch_dtype=torch_dtype,
            trust_remote_code=False,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_layers=lora_layers,
            trainable_layers=trainable_layers,
        )

