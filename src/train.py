from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from datasets import StateIndexMapper, StreetViewDataset, collate_streetview

from models.dinov3 import DinoV3Encoder  # HuggingFace-backed encoder
from models.fusion_transformer import DirectionalFusionTransformer
from models.geo_geussr import GeoGuessrModel, StateHead, GeoHead

from losses import MultiTaskLoss


def seed_everything(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dummy_geo_cells(train_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder geo-cell builder.

    Replace with:
      - S2 cells, or
      - KMeans on 3D Earth coordinates

    Dummy version:
      - single centroid at mean(lat,lon)
      - all samples assigned to cell 0
    """
    import pandas as pd

    df = pd.read_csv(train_csv)
    centroid = np.array(
        [[df["latitude"].mean(), df["longitude"].mean()]], dtype=np.float32
    )
    cell_id = np.zeros(len(df), dtype=np.int64)
    return centroid, cell_id


def main() -> None:
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

    # Fusion
    ap.add_argument("--fusion_layers", type=int, default=2)
    ap.add_argument("--fusion_heads", type=int, default=8)
    ap.add_argument("--fusion_dropout", type=float, default=0.1)

    args = ap.parse_args()

    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    train_csv = data_dir / args.train_csv
    train_images = data_dir / args.train_images

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- State mapping: Kaggle state_idx (non-consecutive) <-> contiguous class_id (0..32)
    state_mapper = StateIndexMapper.from_csv(args.state_mapping)
    num_states = state_mapper.num_states

    # --- DINOv3 preprocessing params (mean/std/size) from HF processor
    pp = DinoV3Encoder.processor_params(args.hf_model_id)

    # Dataset + loader
    ds = StreetViewDataset(
        csv_path=train_csv,
        images_dir=train_images,
        is_train=True,
        state_mapper=state_mapper,
        image_size=pp["image_size"],
        mean=pp["mean"],
        std=pp["std"],
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_streetview,
    )

    # Geo-cells (placeholder)
    centroids_np, cell_ids_np = build_dummy_geo_cells(train_csv)
    np.save(out_dir / "geo_cell_centroids.npy", centroids_np)
    np.save(out_dir / "geo_cell_ids.npy", cell_ids_np)

    # --- Model
    encoder = DinoV3Encoder(model_id=args.hf_model_id, freeze=True).to(device)
    embed_dim = encoder.embed_dim

    fusion = DirectionalFusionTransformer(
        embed_dim=embed_dim,
        num_layers=args.fusion_layers,
        num_heads=args.fusion_heads,
        dropout=args.fusion_dropout,
        use_cls_token=True,
    ).to(device)

    state_head = StateHead(embed_dim=embed_dim, num_states=num_states).to(device)
    geo_head = GeoHead(embed_dim=embed_dim, num_cells=args.num_cells).to(device)

    model = GeoGuessrModel(
        encoder=encoder,
        fusion=fusion,
        state_head=state_head,
        geo_head=geo_head,
    ).to(device)

    # Loss + optimizer
    loss_fn = MultiTaskLoss(
        state_smoothing=0.1,
        lambda_state=1.0,
        lambda_cell=1.0,
        lambda_gps=1.0,
        gps_loss_type="haversine",
    ).to(device)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-2,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        running = {"total": 0.0, "state": 0.0, "cell": 0.0, "gps": 0.0}
        n = 0

        for batch in loader:
            images = batch["images"].to(device)  # (B,4,3,H,W)
            state_class = batch["state_class"].to(device)  # (B,) contiguous
            latlon = batch["latlon"].to(device)  # (B,2)

            # Dummy cell ids (replace with real mapping)
            true_cell = torch.zeros(len(images), dtype=torch.long, device=device)

            with autocast(enabled=(device.type == "cuda")):
                out = model(images)

                # centroid(top cell) + residual (dummy: only one centroid)
                centroid = (
                    torch.tensor(centroids_np[0], device=device, dtype=torch.float32)
                    .view(1, 2)
                    .expand(len(images), 2)
                )
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

        msg = " | ".join(
            [f"{k}: {running[k] / n:.4f}" for k in ["total", "state", "cell", "gps"]]
        )
        print(f"Epoch {epoch}/{args.epochs} - {msg}")

        ckpt_path = out_dir / f"model_epoch_{epoch}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "hf_model_id": args.hf_model_id,
                "state_mapping": args.state_mapping,
                "embed_dim": embed_dim,
                "num_states": num_states,
                "num_cells": args.num_cells,
            },
            ckpt_path,
        )

    print(f"Done. Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()
