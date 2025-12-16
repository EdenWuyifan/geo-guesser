from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from datasets import StateIndexMapper, StreetViewDataset, collate_streetview

from models.dinov3 import DinoV3Encoder  # HuggingFace-backed encoder
from models.fusion_transformer import DirectionalFusionTransformer
from models.geo_geussr import GeoGuessrModel, StateHead, GeoHead

from losses import MultiTaskLoss

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


def main() -> None:
    log_section("🌍 GeoGuessr Training Setup")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--train_csv", type=str, default="train_ground_truth.csv")
    ap.add_argument("--train_images", type=str, default="train_images")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
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
        help="HuggingFace model id for DINOv3.",
    )
    ap.add_argument(
        "--state_mapping",
        type=str,
        default="data/state_mapping.csv",
        help="Path to state_mapping.csv (relative to repo root or absolute).",
    )
    ap.add_argument(
        "--geo_centroids",
        type=str,
        default="data/geo_cell_centroids.npy",
        help="Path to geo_cell_centroids.npy (relative to repo root or absolute).",
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log_step("Device", f"Using CUDA: {torch.cuda.get_device_name(0)}")
        log_step(
            "",
            f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
        )
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
    log_step(
        "Success",
        f"Image size: {pp['image_size']}, Mean: {pp['mean']}, Std: {pp['std']}",
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

    log_step(
        "DataLoader",
        f"Creating DataLoader (batch_size={args.batch_size}, num_workers=1)...",
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_streetview,
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

    # Create mapping from sample_id to row index (for cell_id lookup)
    df_geo = pd.read_csv(train_csv)
    sample_id_to_idx = {int(sid): idx for idx, sid in enumerate(df_geo["sample_id"])}
    log_step(
        "Mapping",
        f"Created sample_id -> row_index mapping for {len(sample_id_to_idx):,} samples",
    )

    log_step("Saving", "Saving geo cell data...")
    np.save(out_dir / "geo_cell_centroids.npy", centroids_np)
    np.save(out_dir / "geo_cell_ids.npy", cell_ids_np)
    log_step("Success", f"Saved to {out_dir}")

    log_section("🧠 Model Initialization")

    log_step("Encoder", f"Loading DINOv3 encoder: {args.hf_model_id}...")
    encoder = DinoV3Encoder(model_id=args.hf_model_id, freeze=True).to(device)
    embed_dim = encoder.embed_dim
    log_step("Success", f"Encoder embed dimension: {embed_dim}")

    log_step("Fusion", "Creating DirectionalFusionTransformer...")
    log_step(
        "",
        f"  Layers: {args.fusion_layers}, Heads: {args.fusion_heads}, Dropout: {args.fusion_dropout}",
    )
    fusion = DirectionalFusionTransformer(
        embed_dim=embed_dim,
        num_layers=args.fusion_layers,
        num_heads=args.fusion_heads,
        dropout=args.fusion_dropout,
        use_cls_token=True,
    ).to(device)

    log_step("State Head", f"Creating StateHead (num_states={num_states})...")
    state_head = StateHead(embed_dim=embed_dim, num_states=num_states).to(device)

    log_step("Geo Head", f"Creating GeoHead (num_cells={len(centroids_np)})...")
    geo_head = GeoHead(embed_dim=embed_dim, num_cells=len(centroids_np)).to(device)

    log_step("Assembling", "Assembling GeoGuessrModel...")
    model = GeoGuessrModel(
        encoder=encoder,
        fusion=fusion,
        state_head=state_head,
        geo_head=geo_head,
    ).to(device)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_step("Success", "Model created")
    log_step("", f"  Total parameters: {total_params:,}")
    log_step("", f"  Trainable parameters: {trainable_params:,}")

    log_section("⚙️  Training Setup")

    log_step("Loss", "Creating MultiTaskLoss...")
    log_step("", "  State smoothing: 0.1, λ_state: 1.0, λ_cell: 1.0, λ_gps: 1.0")
    log_step("", "  GPS loss type: huber")
    loss_fn = MultiTaskLoss(
        state_smoothing=0.1,
        lambda_state=1.0,
        lambda_cell=1.0,
        lambda_gps=1.0,
        gps_loss_type="huber",
    ).to(device)

    log_step(
        "Optimizer", f"Creating AdamW optimizer (lr={args.lr}, weight_decay=1e-2)..."
    )
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-2,
    )

    log_step("Scaler", "Creating GradScaler for mixed precision...")
    scaler = GradScaler(
        "cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")
    )
    if device.type == "cuda":
        log_step("Success", "Mixed precision training enabled")

    log_section("🚀 Training")
    log_step("Starting", f"Training for {args.epochs} epochs on {device}")
    model.train()

    for epoch in range(1, args.epochs + 1):
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'─'*70}{Colors.END}")
        log_step("Epoch", f"{epoch}/{args.epochs}", level="header")
        print(f"{Colors.BOLD}{Colors.CYAN}{'─'*70}{Colors.END}\n")

        running = {"total": 0.0, "state": 0.0, "cell": 0.0, "gps": 0.0}
        n = 0

        for batch_idx, batch in enumerate(loader):
            images = batch["images"].to(device)  # (B,4,3,H,W)
            state_class = batch["state_class"].to(device)  # (B,) contiguous
            latlon = batch["latlon"].to(device)  # (B,2)

            # Get true cell ids from geo_cells using sample_id -> row_index mapping
            sample_ids = batch["sample_id"].cpu().numpy()
            row_indices = np.array([sample_id_to_idx[int(sid)] for sid in sample_ids])
            true_cell = torch.from_numpy(cell_ids_np[row_indices]).to(device)

            dtype = "cuda" if device.type == "cuda" else "cpu"
            with autocast(device_type=dtype, enabled=(device.type == "cuda")):
                out = model(images)

                # Get predicted cell and use its centroid
                pred_cell_indices = out.geo.cell_logits.argmax(dim=-1).cpu().numpy()
                selected_centroids = centroids_np[pred_cell_indices]
                centroid = torch.from_numpy(selected_centroids).to(device).float()
                pred_latlon = centroid + out.geo.residual

                loss_out = loss_fn(
                    state_logits=out.state.logits,
                    cell_logits=out.geo.cell_logits,
                    pred_latlon=pred_latlon,
                    true_state=state_class,
                    true_cell=true_cell,
                    true_latlon=latlon,
                )

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss_out.total).backward()
            scaler.step(optim)
            scaler.update()

            bs = len(images)
            n += bs
            running["total"] += float(loss_out.total.detach().cpu()) * bs
            for k, v in loss_out.parts.items():
                running[k] += float(v.detach().cpu()) * bs

            # Progress update every 100 batches
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
                progress_pct = 100 * (batch_idx + 1) / len(loader)
                current_loss = running["total"] / n
                print(
                    f"{Colors.DIM}[{datetime.now().strftime('%H:%M:%S')}]{Colors.END} "
                    f"{Colors.CYAN}Batch {batch_idx+1:5d}/{len(loader)}{Colors.END} "
                    f"({progress_pct:5.1f}%) | "
                    f"{Colors.YELLOW}Loss: {current_loss:.4f}{Colors.END}",
                    flush=True,
                )

        # Epoch summary
        avg_losses = {k: running[k] / n for k in running.keys()}
        print(f"\n{Colors.BOLD}Epoch {epoch} Summary:{Colors.END}")
        print(
            f"  {Colors.GREEN}Total Loss:{Colors.END} {avg_losses['total']:.6f} | "
            f"{Colors.BLUE}State:{Colors.END} {avg_losses['state']:.6f} | "
            f"{Colors.BLUE}Cell:{Colors.END} {avg_losses['cell']:.6f} | "
            f"{Colors.BLUE}GPS:{Colors.END} {avg_losses['gps']:.6f}"
        )

        log_step("Saving", f"Saving checkpoint for epoch {epoch}...")
        ckpt_path = out_dir / f"model_epoch_{epoch}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "hf_model_id": args.hf_model_id,
                "state_mapping": args.state_mapping,
                "embed_dim": embed_dim,
                "num_states": num_states,
                "num_cells": len(centroids_np),
            },
            ckpt_path,
        )
        log_step("Success", f"Saved to {ckpt_path}", level="success")

    log_section("✅ Training Complete")
    log_step("Complete", f"All checkpoints saved to: {out_dir}", level="success")
    print(f"\n{Colors.BOLD}{Colors.GREEN}Training finished successfully!{Colors.END}\n")


if __name__ == "__main__":
    main()
