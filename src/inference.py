from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import StateIndexMapper, StreetViewDataset, collate_streetview

from models.dinov3 import DinoV3Encoder
from models.fusion_transformer import DirectionalFusionTransformer
from models.geo_geussr import (
    GeoGuessrModel,
    StateHead,
    MixtureGeoHead,
)


def topk_states(probs: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    probs: (B, num_states)
    returns indices: (B, k)
    """
    return torch.topk(probs, k=k, dim=-1).indices


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run inference on test set using trained model checkpoint"
    )
    ap.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    ap.add_argument("--data_dir", type=str, required=True, help="Data directory")
    ap.add_argument("--test_csv", type=str, default="sample_submission.csv")
    ap.add_argument("--test_images", type=str, default="test_images")
    ap.add_argument("--out_csv", type=str, default="submission.csv")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {args.ckpt}")

    # Load checkpoint and extract configuration
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, dict) or "config" not in ckpt:
        raise ValueError(
            "Checkpoint must contain 'config' dict. Please use a checkpoint from training."
        )

    config = ckpt["config"]
    hf_model_id = config["hf_model_id"]
    state_mapping = config["state_mapping"]
    embed_dim = config["embed_dim"]
    num_states = config["num_states"]
    num_cells = config["num_cells"]
    use_mixture = config.get("use_mixture", True)
    num_components = config.get("num_components", 5)
    use_cell_aux = config.get("use_cell_aux", True)
    fusion_layers = config.get("fusion_layers", 2)
    fusion_heads = config.get("fusion_heads", 8)
    fusion_dropout = config.get("fusion_dropout", 0.1)

    # Load centroids (from checkpoint metadata or explicit path)
    if "centroids_path" in ckpt:
        centroids_path = Path(ckpt["centroids_path"])
    else:
        # Fallback: try to find centroids in same directory as checkpoint
        ckpt_dir = Path(args.ckpt).parent
        centroids_path = ckpt_dir / "geo_cell_centroids.npy"
        if not centroids_path.exists():
            raise FileNotFoundError(
                f"Could not find geo cell centroids. Expected at: {centroids_path}"
            )

    print(f"Loading geo cell centroids from: {centroids_path}")
    centroids = np.load(centroids_path)  # (num_cells, 2)

    # Setup data paths
    data_dir = Path(args.data_dir)
    test_csv = data_dir / args.test_csv
    test_images = data_dir / args.test_images

    # Load state mapping
    state_mapper = StateIndexMapper.from_csv(state_mapping)
    assert (
        state_mapper.num_states == num_states
    ), f"State mapping mismatch: {state_mapper.num_states} vs {num_states}"

    # Load DINOv3 preprocessing params
    pp = DinoV3Encoder.processor_params(hf_model_id)
    ds = StreetViewDataset(
        csv_path=test_csv,
        images_dir=test_images,
        is_train=False,
        state_mapper=None,
        image_size=pp["image_size"],
        mean=pp["mean"],
        std=pp["std"],
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_streetview,
    )

    # Rebuild model from checkpoint config
    print("Building model from checkpoint configuration...")
    encoder = DinoV3Encoder(
        model_id=hf_model_id,
        freeze=True,
        use_lora=True,  # Checkpoint should have LoRA if it was trained with it
    )
    assert encoder.embed_dim == embed_dim, "Embed dim mismatch"

    fusion = DirectionalFusionTransformer(
        embed_dim=embed_dim,
        num_layers=fusion_layers,
        num_heads=fusion_heads,
        dropout=fusion_dropout,
        use_global_token=True,
        view_dropout_prob=0.0,  # Disable during inference
    )
    state_head = StateHead(embed_dim=embed_dim, num_states=num_states)

    if use_mixture:
        geo_head = MixtureGeoHead(
            embed_dim=embed_dim,
            num_components=num_components,
            use_cell_aux=use_cell_aux,
            num_cells=num_cells if use_cell_aux else None,
        )
    else:
        from models.geo_geussr import GeoHead

        geo_head = GeoHead(embed_dim=embed_dim, num_cells=num_cells)

    model = GeoGuessrModel(
        encoder=encoder,
        fusion=fusion,
        state_head=state_head,
        geo_head=geo_head,
    ).to(device)

    # Load model weights
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print("Model loaded successfully!")

    # Required columns for submission
    required_cols = [
        "sample_id",
        "image_north",
        "image_east",
        "image_south",
        "image_west",
        "predicted_state_idx_1",
        "predicted_state_idx_2",
        "predicted_state_idx_3",
        "predicted_state_idx_4",
        "predicted_state_idx_5",
        "predicted_latitude",
        "predicted_longitude",
    ]

    # Read template to get sample_id and image paths
    template = pd.read_csv(test_csv)

    all_ids: List[np.ndarray] = []
    all_state_top5_idx: List[np.ndarray] = []
    all_latlon: List[np.ndarray] = []

    for batch in loader:
        images = batch["images"].to(device)  # (B,4,3,H,W)
        sample_id = batch["sample_id"].cpu().numpy()  # (B,)

        out = model(images)

        # State predictions: top-5 over contiguous class IDs (0..32)
        top5_class = topk_states(out.state.probs, k=5).cpu().numpy()  # (B,5)

        # Map contiguous class IDs back to Kaggle/original state_idx values
        top5_state_idx = np.empty_like(top5_class, dtype=np.int64)
        for i in range(top5_class.shape[0]):
            for j in range(top5_class.shape[1]):
                top5_state_idx[i, j] = state_mapper.to_state_idx(int(top5_class[i, j]))

        # Geo predictions: handle mixture model vs old model
        if use_mixture:
            # Mixture model: use weighted mean of top component
            # Get top component by weight
            top_component = out.geo.weights.argmax(dim=-1)  # (B,)
            pred_latlon = (
                out.geo.means[torch.arange(len(top_component)), top_component]
                .cpu()
                .numpy()
            )  # (B, 2)
        else:
            # Old model: centroid(top cell) + residual
            top_cell = out.geo.cell_probs.argmax(dim=-1).cpu().numpy()  # (B,)
            centroid = torch.tensor(
                centroids[top_cell], device=device, dtype=torch.float32
            )  # (B,2)
            pred_latlon = (centroid + out.geo.residual).cpu().numpy()  # (B,2)

        all_ids.append(sample_id)
        all_state_top5_idx.append(top5_state_idx)
        all_latlon.append(pred_latlon)

    ids = np.concatenate(all_ids, axis=0)
    top5_idx = np.concatenate(all_state_top5_idx, axis=0)
    latlon = np.concatenate(all_latlon, axis=0)

    # Create submission dataframe with only required columns
    # Start with template data (sample_id and image paths)
    submission = template[
        ["sample_id", "image_north", "image_east", "image_south", "image_west"]
    ].copy()

    # Initialize prediction columns
    for col in [
        "predicted_state_idx_1",
        "predicted_state_idx_2",
        "predicted_state_idx_3",
        "predicted_state_idx_4",
        "predicted_state_idx_5",
        "predicted_latitude",
        "predicted_longitude",
    ]:
        submission[col] = np.nan

    # Write predictions back by sample_id alignment
    id_to_row = {int(sid): i for i, sid in enumerate(submission["sample_id"].values)}
    for sid, t5, ll in zip(ids, top5_idx, latlon):
        r = id_to_row[int(sid)]
        # Ensure state IDs are integers
        submission.loc[r, "predicted_state_idx_1"] = int(t5[0])
        submission.loc[r, "predicted_state_idx_2"] = int(t5[1])
        submission.loc[r, "predicted_state_idx_3"] = int(t5[2])
        submission.loc[r, "predicted_state_idx_4"] = int(t5[3])
        submission.loc[r, "predicted_state_idx_5"] = int(t5[4])
        submission.loc[r, "predicted_latitude"] = float(ll[0])
        submission.loc[r, "predicted_longitude"] = float(ll[1])

    # Ensure state ID columns are integers (convert any remaining NaN/invalid to 0)
    state_cols = [
        "predicted_state_idx_1",
        "predicted_state_idx_2",
        "predicted_state_idx_3",
        "predicted_state_idx_4",
        "predicted_state_idx_5",
    ]
    for col in state_cols:
        submission[col] = (
            pd.to_numeric(submission[col], errors="coerce").fillna(0).astype(int)
        )

    # Reorder columns to match required order
    submission = submission[required_cols]

    out_path = Path(args.out_csv)
    submission.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path}")


if __name__ == "__main__":
    main()
