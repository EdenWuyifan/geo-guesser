from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) wrapper for a linear layer.

    Wraps an existing nn.Linear layer and adds low-rank adaptation:
    W' = W + B @ A, where B is (out_features, rank) and A is (rank, in_features)
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            linear: The original linear layer to wrap
            rank: LoRA rank (r)
            alpha: LoRA scaling factor (typically rank * 2)
            dropout: Dropout probability for LoRA adapters
        """
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False

        # LoRA adapters: B (down) and A (up)
        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize: A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=torch.sqrt(torch.tensor(5.0)))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: W @ x + (B @ A) @ x * scaling
        """
        # Original output
        out = self.linear(x)

        # LoRA adaptation
        lora_out = self.lora_dropout(x) @ self.lora_A.T  # (..., rank)
        lora_out = lora_out @ self.lora_B.T  # (..., out_features)
        out = out + lora_out * self.scaling

        return out


def add_lora_to_attention(
    attn_module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list[str] | None = None,
) -> None:
    """
    Add LoRA adapters to attention module's q and v projections.

    Args:
        attn_module: The attention module (typically has .qkv or .query/.key/.value)
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: List of projection names to target (default: ['q', 'v'])
    """
    if target_modules is None:
        target_modules = ["q", "v"]

    # Handle different attention architectures
    # DINOv3 typically uses .qkv (combined) or separate .query, .key, .value
    if hasattr(attn_module, "qkv"):
        # Combined qkv projection - need to split it
        qkv = attn_module.qkv
        dim = qkv.out_features // 3
        # Create separate projections
        q_proj = nn.Linear(qkv.in_features, dim, bias=qkv.bias is not None)
        k_proj = nn.Linear(qkv.in_features, dim, bias=qkv.bias is not None)
        v_proj = nn.Linear(qkv.in_features, dim, bias=qkv.bias is not None)

        # Copy weights from qkv (assuming qkv is [Q; K; V] concatenated)
        with torch.no_grad():
            q_proj.weight.copy_(qkv.weight[:dim])
            k_proj.weight.copy_(qkv.weight[dim : 2 * dim])
            v_proj.weight.copy_(qkv.weight[2 * dim :])
            if qkv.bias is not None:
                q_proj.bias.copy_(qkv.bias[:dim])
                k_proj.bias.copy_(qkv.bias[dim : 2 * dim])
                v_proj.bias.copy_(qkv.bias[2 * dim :])

        # Freeze base projections by default; LoRA layers hold trainable params
        for p in q_proj.parameters():
            p.requires_grad = False
        for p in k_proj.parameters():
            p.requires_grad = False
        for p in v_proj.parameters():
            p.requires_grad = False

        # Replace with LoRA versions for q and v
        if "q" in target_modules:
            attn_module.q_proj = LoRALinear(q_proj, rank=rank, alpha=alpha)
        else:
            attn_module.q_proj = q_proj

        attn_module.k_proj = k_proj  # Keep k without LoRA

        if "v" in target_modules:
            attn_module.v_proj = LoRALinear(v_proj, rank=rank, alpha=alpha)
        else:
            attn_module.v_proj = v_proj

        # Remove original qkv
        del attn_module.qkv

    elif hasattr(attn_module, "query") and hasattr(attn_module, "value"):
        # Separate query, key, value projections
        if "q" in target_modules:
            attn_module.query = LoRALinear(attn_module.query, rank=rank, alpha=alpha)
        if "v" in target_modules:
            attn_module.value = LoRALinear(attn_module.value, rank=rank, alpha=alpha)
    elif hasattr(attn_module, "q_proj") and hasattr(attn_module, "v_proj"):
        # Already has separate projections
        if "q" in target_modules:
            attn_module.q_proj = LoRALinear(attn_module.q_proj, rank=rank, alpha=alpha)
        if "v" in target_modules:
            attn_module.v_proj = LoRALinear(attn_module.v_proj, rank=rank, alpha=alpha)
    else:
        raise ValueError(
            f"Unsupported attention module structure: {type(attn_module)}. "
            "Expected .qkv, .query/.value, or .q_proj/.v_proj"
        )
