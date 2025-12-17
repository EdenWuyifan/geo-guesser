from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from datasets import StateIndexMapper, StreetViewDataset, collate_streetview

from models.dinov3 import DinoV3Encoder  # HuggingFace-backed encoder
from models.fusion_transformer import DirectionalFusionTransformer
from models.geo_geussr import (
    GeoGuessrModel,
    StateHead,
    MixtureGeoHead,
)

from losses import MixtureMultiTaskLoss
from self_supervised_losses import SelfSupervisedLoss

from geo_cell import build_geo_cells


# ANSI color codes for pretty printing
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
    DIM = "\033[2m"


def log_step(step: str, message: str = "", level: str = "info") -> None:
    """Print a formatted log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    colors = {
        "header": Colors.HEADER + Colors.BOLD,
        "step": Colors.CYAN + Colors.BOLD,
        "info": Colors.BLUE,
        "success": Colors.GREEN,
        "warning": Colors.YELLOW,
        "error": Colors.RED,
    }
    color = colors.get(level, colors["info"])
    reset = Colors.END

    if message:
        print(
            f"{Colors.DIM}[{timestamp}]{reset} {color}{step:15s}{reset} {message}",
            flush=True,
        )
    else:
        print(f"{Colors.DIM}[{timestamp}]{reset} {color}{step:15s}{reset}", flush=True)


def log_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{title:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}\n")


def seed_everything(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    """Find the latest checkpoint file in out_dir matching model_epoch_*.pt pattern."""
    checkpoints = list(out_dir.glob("model_epoch_*.pt"))
    if not checkpoints:
        return None

    # Extract epoch numbers and find the latest
    def extract_epoch(path: Path) -> int:
        match = re.search(r"model_epoch_(\d+)\.pt", path.name)
        return int(match.group(1)) if match else -1

    latest = max(checkpoints, key=extract_epoch)
    return latest


def main() -> None:
    log_section("🌍 GeoGuessr Training Setup")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--train_csv", type=str, default="train_ground_truth.csv")
    ap.add_argument("--train_images", type=str, default="train_images")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size. Increase for higher GPU utilization (if memory allows).",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_cells", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--seed", type=int, default=42)

    # DINOv3 / mapping
    ap.add_argument(
        "--hf_model_id",
        type=str,
        default="facebook/dinov3-vit7b16-pretrain-lvd1689m",
        help="HuggingFace model id for DINOv3. "
        "Faster alternatives: 'facebook/dinov2-vitb14' (base), "
        "'facebook/dinov2-vits14' (small, fastest)",
    )
    ap.add_argument(
        "--compile_model",
        type=str,
        default="none",
        choices=["none", "full", "heads"],
        help="Compilation strategy: 'none' (default, recommended for debugging), "
        "'full' (compile entire model), 'heads' (compile only trainable heads)",
    )
    ap.add_argument(
        "--state_mapping",
        type=str,
        default="data/state_mapping.csv",
        help="Path to state_mapping.csv (relative to repo root or absolute).",
    )
    ap.add_argument(
        "--geo_method",
        type=str,
        default="kmeans3d",
        choices=["s2", "kmeans3d"],
        help="Method for building geo cells: 's2' or 'kmeans3d'.",
    )

    # Fusion
    ap.add_argument("--fusion_layers", type=int, default=2)
    ap.add_argument("--fusion_heads", type=int, default=8)
    ap.add_argument("--fusion_dropout", type=float, default=0.1)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers (increase for faster data loading). "
        "Recommended: 4-8 for most systems, higher if you have many CPU cores.",
    )
    ap.add_argument(
        "--prefetch_factor",
        type=int,
        default=8,
        help="DataLoader prefetch factor (higher = more prefetching). "
        "Higher values keep GPU fed with data.",
    )
    ap.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps. "
        "Useful to simulate larger batch sizes when limited by GPU memory.",
    )

    args = ap.parse_args()

    log_step("Configuration", "Parsing arguments...")
    log_step("", f"  Data directory: {args.data_dir}")
    log_step("", f"  Training CSV: {args.train_csv}")
    log_step("", f"  Images directory: {args.train_images}")
    log_step("", f"  Epochs: {args.epochs}")
    log_step("", f"  Batch size: {args.batch_size}")
    log_step("", f"  Learning rate: {args.lr}")
    log_step("", f"  Number of cells: {args.num_cells}")
    log_step("", f"  Geo method: {args.geo_method}")
    log_step("", f"  Output directory: {args.out_dir}")
    log_step("", f"  Seed: {args.seed}")

    log_step("Seeding", f"Setting random seed to {args.seed}...")
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    train_csv = data_dir / args.train_csv
    train_images = data_dir / args.train_images

    log_step("Directories", "Setting up paths...")
    log_step("", f"  Training CSV: {train_csv}")
    log_step("", f"  Images: {train_images}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_step("Output", f"Checkpoints will be saved to: {out_dir}")

    # Check for existing checkpoints to resume from
    resume_checkpoint = find_latest_checkpoint(out_dir)
    start_epoch = 1
    if resume_checkpoint:
        log_step(
            "Resume",
            f"Found checkpoint: {resume_checkpoint.name}",
            level="success",
        )
    else:
        log_step("Resume", "No existing checkpoint found, starting from scratch")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log_step("Device", f"Using CUDA: {torch.cuda.get_device_name(0)}")
        gpu_props = torch.cuda.get_device_properties(0)
        log_step(
            "",
            f"  GPU Memory: {gpu_props.total_memory / 1e9:.2f} GB",
        )
        # Enable CUDA optimizations for better GPU utilization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        log_step(
            "CUDA",
            "Enabled cudnn.benchmark and high-precision matmul for optimized performance",
        )

        # Set memory allocation strategy
        torch.cuda.empty_cache()
    else:
        log_step("Device", "Using CPU")

    log_section("📊 Data Loading")

    # --- State mapping: Kaggle state_idx (non-consecutive) <-> contiguous class_id (0..32)
    log_step("State Mapping", f"Loading from {args.state_mapping}...")
    state_mapper = StateIndexMapper.from_csv(args.state_mapping)
    num_states = state_mapper.num_states
    log_step("Success", f"Loaded {num_states} states")

    # --- DINOv3 preprocessing params (mean/std/size) from HF processor
    log_step("DINOv3 Config", f"Fetching processor params for {args.hf_model_id}...")
    pp = DinoV3Encoder.processor_params(args.hf_model_id)
    image_size = pp["image_size"]
    log_step(
        "Success",
        f"Image size: {image_size}x{image_size}, Mean: {pp['mean']}, Std: {pp['std']}",
    )
    if image_size > 224:
        log_step(
            "Note",
            f"Large image size ({image_size}x{image_size}) will slow training. "
            "Consider smaller sizes (224x224 or 256x256) if acceptable.",
            level="info",
        )

    # Dataset + loader
    log_step("Dataset", f"Creating StreetViewDataset from {train_csv}...")
    ds = StreetViewDataset(
        csv_path=train_csv,
        images_dir=train_images,
        is_train=True,
        state_mapper=state_mapper,
        image_size=pp["image_size"],
        mean=pp["mean"],
        std=pp["std"],
    )
    log_step("Success", f"Dataset size: {len(ds):,} samples")

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    log_step(
        "DataLoader",
        f"Creating DataLoader (batch_size={args.batch_size}, "
        f"num_workers={args.num_workers}, prefetch_factor={args.prefetch_factor})...",
    )
    if args.gradient_accumulation_steps > 1:
        log_step(
            "",
            f"Gradient accumulation: {args.gradient_accumulation_steps} steps "
            f"(effective batch size: {effective_batch_size})",
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_streetview,
        persistent_workers=(args.num_workers > 0),
    )
    log_step("Success", f"Batches per epoch: {len(loader):,}")

    log_section("🗺️  Geo-Cell Building")
    log_step("Building", f"Creating geo cells using method: {args.geo_method}...")
    geo_cells = build_geo_cells(
        train_csv, method=args.geo_method, num_cells=args.num_cells
    )
    centroids_np = geo_cells.centroids_latlon
    cell_ids_np = geo_cells.cell_ids
    log_step("Success", f"Created {len(centroids_np):,} geo cells")
    log_step("", f"  Method: {geo_cells.meta.get('method', 'unknown')}")
    if "s2_level" in geo_cells.meta:
        log_step("", f"  S2 Level: {geo_cells.meta['s2_level']}")

    # Pre-allocate cell_ids tensor on GPU for faster batch lookup (using row_idx from dataset)
    cell_ids_tensor = torch.from_numpy(cell_ids_np).to(device)
    log_step("Pre-allocation", "Pre-allocated cell_ids tensor on GPU")

    log_step("Saving", "Saving geo cell data...")
    np.save(out_dir / "geo_cell_centroids.npy", centroids_np)
    np.save(out_dir / "geo_cell_ids.npy", cell_ids_np)
    log_step("Success", f"Saved to {out_dir}")

    log_section("🧠 Model Initialization")

    log_step("Encoder", f"Loading DINOv3 encoder: {args.hf_model_id}...")
    # Use bfloat16 for large models to save memory and potentially speed up on modern GPUs
    torch_dtype = torch.bfloat16 if device.type == "cuda" else None

    # Enable LoRA on last 4 blocks by default
    lora_rank = 8
    lora_alpha = 16.0
    lora_layers = 4
    log_step(
        "LoRA",
        f"Adding LoRA adapters (rank={lora_rank}, alpha={lora_alpha}, layers={lora_layers})...",
    )
    encoder = DinoV3Encoder(
        model_id=args.hf_model_id,
        freeze=True,  # Freeze base weights, only LoRA params trainable
        torch_dtype=torch_dtype,
        use_lora=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_layers=lora_layers,
    ).to(device)
    embed_dim = encoder.embed_dim
    log_step("Success", f"Encoder embed dimension: {embed_dim}")

    # Warn if using a very large model
    if "7b" in args.hf_model_id.lower() or "giant" in args.hf_model_id.lower():
        log_step(
            "Warning",
            "Using a large model (7B+). Consider using a smaller variant for faster training:",
            level="warning",
        )
        log_step(
            "",
            "  - 'facebook/dinov2-vitb14' (base, ~86M params, ~10-20x faster)",
            level="warning",
        )
        log_step(
            "",
            "  - 'facebook/dinov2-vits14' (small, ~22M params, ~30-50x faster)",
            level="warning",
        )

    log_step("Fusion", "Creating DirectionalFusionTransformer...")
    log_step(
        "",
        f"  Layers: {args.fusion_layers}, Heads: {args.fusion_heads}, Dropout: {args.fusion_dropout}",
    )
    # Enable view dropout for training robustness (drop 1-2 views randomly)
    view_dropout_prob = 0.15  # 15% chance to drop views per sample
    log_step("", f"  View dropout: {view_dropout_prob:.1%} (for training robustness)")
    fusion = DirectionalFusionTransformer(
        embed_dim=embed_dim,
        num_layers=args.fusion_layers,
        num_heads=args.fusion_heads,
        dropout=args.fusion_dropout,
        use_global_token=True,
        view_dropout_prob=view_dropout_prob,
    ).to(device)

    log_step("State Head", f"Creating StateHead (num_states={num_states})...")
    state_head = StateHead(embed_dim=embed_dim, num_states=num_states).to(device)

    # Default: use mixture-of-experts regression (5 components, with cell aux loss)
    num_components = 5
    use_cell_aux = True
    log_step(
        "Geo Head",
        f"Creating MixtureGeoHead (num_components={num_components}, "
        f"use_cell_aux={use_cell_aux})...",
    )
    geo_head = MixtureGeoHead(
        embed_dim=embed_dim,
        num_components=num_components,
        use_cell_aux=use_cell_aux,
        num_cells=len(centroids_np),
    ).to(device)

    log_step("Assembling", "Assembling GeoGuessrModel...")
    model = GeoGuessrModel(
        encoder=encoder,
        fusion=fusion,
        state_head=state_head,
        geo_head=geo_head,
    ).to(device)

    # Convert model to channels-last memory format for better GPU utilization (Ampere+)
    if device.type == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
            log_step("Memory", "Converted model to channels-last memory format")
        except Exception as e:
            log_step(
                "Warning",
                f"Failed to convert to channels-last: {e}",
                level="warning",
            )

    # Compile model selectively (PyTorch 2.0+)
    if args.compile_model == "full":
        try:
            log_step("Compiling", "Compiling entire model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            log_step("Success", "Model compiled for faster inference")
        except Exception as e:
            log_step(
                "Warning",
                f"torch.compile() failed (may not be available): {e}",
                level="warning",
            )
    elif args.compile_model == "heads":
        try:
            log_step(
                "Compiling",
                "Compiling only trainable heads with torch.compile()...",
            )
            # Compile only the trainable components (heads and fusion)
            model.fusion = torch.compile(model.fusion, mode="reduce-overhead")
            model.state_head = torch.compile(model.state_head, mode="reduce-overhead")
            model.geo_head = torch.compile(model.geo_head, mode="reduce-overhead")
            log_step("Success", "Trainable heads compiled for faster inference")
        except Exception as e:
            log_step(
                "Warning",
                f"torch.compile() on heads failed: {e}",
                level="warning",
            )
    else:
        log_step(
            "Compile",
            "Model compilation disabled (use --compile_model full/heads to enable)",
        )

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_step("Success", "Model created")
    log_step("", f"  Total parameters: {total_params:,}")
    log_step("", f"  Trainable parameters: {trainable_params:,}")

    # Estimate model size in GB
    encoder_params = sum(p.numel() for p in encoder.parameters())
    model_size_gb = total_params * 4 / (1024**3)  # Assuming float32 (4 bytes)
    log_step(
        "",
        f"  Encoder parameters: {encoder_params:,} ({encoder_params * 4 / (1024**3):.2f} GB)",
    )
    log_step("", f"  Estimated model size: {model_size_gb:.2f} GB")

    log_section("⚙️  Training Setup")

    log_step("Loss", "Creating MixtureMultiTaskLoss...")
    log_step(
        "",
        f"  State smoothing: 0.1, λ_state: 1.0, λ_gps: 1.0, "
        f"use_cell_aux: {use_cell_aux}",
    )
    log_step("", "  λ_cell: 0.5 (auxiliary loss)")
    loss_fn = MixtureMultiTaskLoss(
        state_smoothing=0.1,
        lambda_state=1.0,
        lambda_cell=0.5,
        lambda_gps=1.0,
        use_cell_aux=use_cell_aux,
    ).to(device)

    # Self-supervised losses for better geographic embedding space
    log_step("Self-Supervised", "Creating SelfSupervisedLoss...")
    log_step("", "  View consistency: λ=0.1 (4 views should be similar)")
    log_step("", "  Geo-distance contrastive: λ=0.1 (nearby=positive, far=negative)")
    self_sup_loss_fn = SelfSupervisedLoss(
        lambda_view_consistency=0.1,
        lambda_geo_distance=0.1,
        view_temperature=0.07,
        geo_temperature=0.07,
        positive_threshold_km=100.0,
        negative_threshold_km=1000.0,
    ).to(device)

    log_step("Optimizer", "Setting up layer-wise learning rate decay (LLRD)...")

    # Layer-wise learning rate decay strategy:
    # - Heads & Fusion: high LR (base_lr)
    # - Last blocks (LoRA): medium LR (base_lr * 0.1)
    # - Earlier blocks: frozen (LR = 0, but handled by freeze=True)

    base_lr = args.lr
    lora_lr = base_lr * 0.1  # 10x smaller for LoRA adapters

    # Group parameters by component
    param_groups = []

    # Heads and fusion: high LR
    head_fusion_params = []
    if hasattr(model, "state_head"):
        head_fusion_params.extend(
            [p for p in model.state_head.parameters() if p.requires_grad]
        )
    if hasattr(model, "geo_head"):
        head_fusion_params.extend(
            [p for p in model.geo_head.parameters() if p.requires_grad]
        )
    if hasattr(model, "fusion"):
        head_fusion_params.extend(
            [p for p in model.fusion.parameters() if p.requires_grad]
        )

    if head_fusion_params:
        param_groups.append(
            {
                "params": head_fusion_params,
                "lr": base_lr,
                "name": "heads_fusion",
            }
        )
        log_step(
            "", f"  Heads & Fusion: {len(head_fusion_params)} params @ LR={base_lr:.2e}"
        )

    # LoRA adapters: medium LR
    lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            lora_params.append(param)

    if lora_params:
        param_groups.append(
            {
                "params": lora_params,
                "lr": lora_lr,
                "name": "lora_adapters",
            }
        )
        log_step("", f"  LoRA adapters: {len(lora_params)} params @ LR={lora_lr:.2e}")

    # Create optimizer with parameter groups
    optim = torch.optim.AdamW(
        param_groups,
        weight_decay=1e-2,
    )
    log_step("Success", f"Optimizer created with {len(param_groups)} parameter groups")

    log_step("Scaler", "Creating GradScaler for mixed precision...")
    scaler = GradScaler(
        "cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")
    )
    if device.type == "cuda":
        log_step("Success", "Mixed precision training enabled")

    log_section("🚀 Training")

    # Load checkpoint if resuming
    if resume_checkpoint:
        log_step("Loading", f"Loading checkpoint from {resume_checkpoint.name}...")
        ckpt = torch.load(resume_checkpoint, map_location=device)

        # Verify checkpoint compatibility
        ckpt_config = ckpt.get("config", {})
        if not ckpt_config:
            # Legacy checkpoint format
            ckpt_config = {
                "embed_dim": ckpt.get("embed_dim"),
                "num_states": ckpt.get("num_states"),
                "use_mixture": ckpt.get("use_mixture", True),
                "num_components": ckpt.get("num_components", num_components),
                "use_cell_aux": ckpt.get("use_cell_aux", use_cell_aux),
            }

        ckpt_embed_dim = ckpt_config.get("embed_dim")
        ckpt_num_states = ckpt_config.get("num_states")
        ckpt_use_mixture = ckpt_config.get("use_mixture", True)

        if (
            ckpt_embed_dim != embed_dim
            or ckpt_num_states != num_states
            or ckpt_use_mixture is not True
        ):
            log_step(
                "Error",
                f"Checkpoint incompatible: embed_dim={ckpt_embed_dim} vs {embed_dim}, "
                f"num_states={ckpt_num_states} vs {num_states}, "
                f"use_mixture={ckpt_use_mixture} vs True",
                level="error",
            )
            raise ValueError("Checkpoint configuration mismatch")

        ckpt_num_components = ckpt_config.get("num_components", num_components)
        ckpt_use_cell_aux = ckpt_config.get("use_cell_aux", use_cell_aux)
        if ckpt_num_components != num_components or ckpt_use_cell_aux != use_cell_aux:
            log_step(
                "Error",
                f"Mixture model mismatch: num_components={ckpt_num_components} vs {num_components}, "
                f"use_cell_aux={ckpt_use_cell_aux} vs {use_cell_aux}",
                level="error",
            )
            raise ValueError("Checkpoint mixture configuration mismatch")

        # Load model state
        model.load_state_dict(ckpt["model"], strict=True)

        # Load optimizer and scaler state if available
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
            log_step("Success", "Loaded optimizer state")
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
            log_step("Success", "Loaded scaler state")

        # Determine starting epoch
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            log_step("Success", f"Resuming from epoch {start_epoch}")
        else:
            # Fallback: extract epoch from filename
            match = re.search(r"model_epoch_(\d+)\.pt", resume_checkpoint.name)
            if match:
                start_epoch = int(match.group(1)) + 1
                log_step(
                    "Success",
                    f"Resuming from epoch {start_epoch} (extracted from filename)",
                )
            else:
                log_step(
                    "Warning",
                    "Could not determine resume epoch, starting from epoch 1",
                    level="warning",
                )
                start_epoch = 1

        if start_epoch > args.epochs:
            log_step(
                "Warning",
                f"Checkpoint epoch {start_epoch-1} >= target epochs {args.epochs}, nothing to train",
                level="warning",
            )
    else:
        log_step("Starting", f"Training for {args.epochs} epochs on {device}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        log_step(
            "Memory",
            f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
        )
    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'─'*70}{Colors.END}")
        log_step("Epoch", f"{epoch}/{args.epochs}", level="header")
        print(f"{Colors.BOLD}{Colors.CYAN}{'─'*70}{Colors.END}\n")

        # Use GPU tensors for accumulation to avoid CPU sync on every batch
        running_total = torch.zeros((), device=device, dtype=torch.float32)
        running_state = torch.zeros((), device=device, dtype=torch.float32)
        running_cell = torch.zeros((), device=device, dtype=torch.float32)
        running_gps = torch.zeros((), device=device, dtype=torch.float32)
        seen = torch.zeros((), device=device, dtype=torch.float32)
        optim.zero_grad(set_to_none=True)  # Initialize gradients at start of epoch

        for batch_idx, batch in enumerate(loader):
            # Transfer to device with channels-last if supported
            images = batch["images"].to(device, non_blocking=True)  # (B,4,3,H,W)
            if device.type == "cuda":
                try:
                    images = images.to(memory_format=torch.channels_last)
                except Exception:
                    pass  # Fallback to contiguous if not supported
            state_class = batch["state_class"].to(device, non_blocking=True)  # (B,)
            latlon = batch["latlon"].to(device, non_blocking=True)  # (B,2)

            # Get true cell ids using row_idx directly (no Python loop, no CPU sync)
            row_idx = batch["row_idx"].to(device, non_blocking=True)
            true_cell = cell_ids_tensor[row_idx]

            dtype = "cuda" if device.type == "cuda" else "cpu"
            with autocast(device_type=dtype, enabled=(device.type == "cuda")):
                out = model(images)

                # Supervised loss: mixture-of-experts
                loss_out = loss_fn(
                    state_logits=out.state.logits,
                    means=out.geo.means,
                    covariances=out.geo.covariances,
                    weights=out.geo.weights,
                    true_state=state_class,
                    true_latlon=latlon,
                    cell_logits=out.geo.cell_logits,
                    true_cell=true_cell,
                )

                # Self-supervised losses: view consistency + geo-distance contrastive
                # Use view tokens from model output (no extra forward pass needed)
                self_sup_losses = self_sup_loss_fn(
                    view_tokens=out.view_tokens,  # (B, V=4, D)
                    fused_embeddings=out.fused,  # (B, D)
                    latlon=latlon,  # (B, 2)
                )

                # Combine supervised and self-supervised losses
                total_loss = loss_out.total + self_sup_losses["total"]

            # Scale loss by accumulation steps for correct averaging
            scaled_loss = total_loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(loader):
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            bs = len(images)
            # Accumulate on GPU (no CPU sync until logging) - massive speedup
            with torch.no_grad():
                running_total += total_loss.detach() * bs
                running_state += loss_out.parts["state"].detach() * bs
                if "cell" in loss_out.parts:
                    running_cell += loss_out.parts["cell"].detach() * bs
                running_gps += loss_out.parts["gps"].detach() * bs
                seen += bs

            # Progress update every 100 batches (only sync GPU->CPU here)
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
                progress_pct = 100 * (batch_idx + 1) / len(loader)
                current_loss = (
                    running_total / seen
                ).item()  # Single GPU sync per 100 steps
                print(
                    f"{Colors.DIM}[{datetime.now().strftime('%H:%M:%S')}]{Colors.END} "
                    f"{Colors.CYAN}Batch {batch_idx+1:5d}/{len(loader)}{Colors.END} "
                    f"({progress_pct:5.1f}%) | "
                    f"{Colors.YELLOW}Loss: {current_loss:.4f}{Colors.END}",
                    flush=True,
                )

        # Epoch summary (sync GPU->CPU only once at epoch end)
        avg_total = (running_total / seen).item()
        avg_state = (running_state / seen).item()
        avg_cell = (running_cell / seen).item()
        avg_gps = (running_gps / seen).item()
        print(f"\n{Colors.BOLD}Epoch {epoch} Summary:{Colors.END}")
        print(
            f"  {Colors.GREEN}Total Loss:{Colors.END} {avg_total:.6f} | "
            f"{Colors.BLUE}State:{Colors.END} {avg_state:.6f} | "
            f"{Colors.BLUE}Cell (aux):{Colors.END} {avg_cell:.6f} | "
            f"{Colors.BLUE}GPS (NLL):{Colors.END} {avg_gps:.6f}"
        )
        print(
            f"  {Colors.CYAN}Note:{Colors.END} Total includes self-supervised losses "
            f"(view consistency + geo-distance contrastive)"
        )

        log_step("Saving", f"Saving checkpoint for epoch {epoch}...")
        ckpt_path = out_dir / f"model_epoch_{epoch}.pt"

        # Save geo cell centroids alongside checkpoint
        centroids_path = out_dir / f"geo_cell_centroids_epoch_{epoch}.npy"
        np.save(centroids_path, centroids_np)

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scaler": scaler.state_dict(),
                "config": {
                    "hf_model_id": args.hf_model_id,
                    "state_mapping": args.state_mapping,
                    "embed_dim": embed_dim,
                    "num_states": num_states,
                    "num_cells": len(centroids_np),
                    "use_mixture": True,
                    "num_components": num_components,
                    "use_cell_aux": use_cell_aux,
                    "geo_method": args.geo_method,
                    "fusion_layers": args.fusion_layers,
                    "fusion_heads": args.fusion_heads,
                    "fusion_dropout": args.fusion_dropout,
                },
                "centroids_path": str(centroids_path),
            },
            ckpt_path,
        )
        log_step("Success", f"Saved to {ckpt_path}", level="success")

    log_section("✅ Training Complete")
    log_step("Complete", f"All checkpoints saved to: {out_dir}", level="success")
    print(f"\n{Colors.BOLD}{Colors.GREEN}Training finished successfully!{Colors.END}\n")


if __name__ == "__main__":
    main()
