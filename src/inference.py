from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import StateIndexMapper, StreetViewDataset, collate_streetview

from models.dinov3 import DinoV3Encoder  # HuggingFace-backed encoder
from models.fusion_transformer import DirectionalFusionTransformer
from models.geo_geussr import GeoGuessrModel, StateHead, GeoHead


def topk_states(probs: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    probs: (B, num_states)
    returns indices: (B, k)
    """
    return torch.topk(probs, k=k, dim=-1).indices


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--test_csv", type=str, default="sample_submission.csv")
    ap.add_argument("--test_images", type=str, default="test_images")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument(
        "--geo_centroids", type=str, required=True
    )  # .npy saved from training
    ap.add_argument("--out_csv", type=str, default="submission.csv")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_cells", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")

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

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    test_csv = data_dir / args.test_csv
    test_images = data_dir / args.test_images

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load state mapping (original Kaggle state_idx <-> contiguous class_id)
    state_mapper = StateIndexMapper.from_csv(args.state_mapping)
    num_states = state_mapper.num_states

    # Load centroids
    centroids = np.load(args.geo_centroids)  # (num_cells,2)

    # Load DINOv3 preprocessing params and build dataset accordingly
    pp = DinoV3Encoder.processor_params(args.hf_model_id)
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
        pin_memory=True,
        collate_fn=collate_streetview,
    )

    # Rebuild model (embed_dim inferred from HF config inside DinoV3Encoder)
    encoder = DinoV3Encoder(model_id=args.hf_model_id, freeze=True)
    embed_dim = encoder.embed_dim

    fusion = DirectionalFusionTransformer(
        embed_dim=embed_dim,
        num_layers=2,
        num_heads=8,
        dropout=0.0,
        use_cls_token=True,
    )
    state_head = StateHead(embed_dim=embed_dim, num_states=num_states)
    geo_head = GeoHead(embed_dim=embed_dim, num_cells=args.num_cells)

    model = GeoGuessrModel(
        encoder=encoder,
        fusion=fusion,
        state_head=state_head,
        geo_head=geo_head,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    # ckpt can be either {"model": state_dict, ...} or raw state_dict
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Prepare output rows using the provided template
    template = pd.read_csv(test_csv)
    for col in [
        "predicted_state_idx_1",
        "predicted_state_idx_2",
        "predicted_state_idx_3",
        "predicted_state_idx_4",
        "predicted_state_idx_5",
        "predicted_latitude",
        "predicted_longitude",
    ]:
        if col not in template.columns:
            template[col] = np.nan

    all_ids: List[np.ndarray] = []
    all_state_top5_idx: List[np.ndarray] = []
    all_latlon: List[np.ndarray] = []

    for batch in loader:
        images = batch["images"].to(device)  # (B,4,3,H,W)
        sample_id = batch["sample_id"].cpu().numpy()  # (B,)

        out = model(images)

        # state top-5 over contiguous class IDs (0..32)
        top5_class = topk_states(out.state.probs, k=5).cpu().numpy()  # (B,5)

        # map contiguous class IDs back to Kaggle/original state_idx values
        top5_state_idx = np.empty_like(top5_class, dtype=np.int64)
        for i in range(top5_class.shape[0]):
            for j in range(top5_class.shape[1]):
                top5_state_idx[i, j] = state_mapper.to_state_idx(int(top5_class[i, j]))

        # geo: centroid(top cell) + residual
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

    # Write back by sample_id alignment
    id_to_row = {int(sid): i for i, sid in enumerate(template["sample_id"].values)}
    for sid, t5, ll in zip(ids, top5_idx, latlon):
        r = id_to_row[int(sid)]
        template.loc[r, "predicted_state_idx_1"] = int(t5[0])
        template.loc[r, "predicted_state_idx_2"] = int(t5[1])
        template.loc[r, "predicted_state_idx_3"] = int(t5[2])
        template.loc[r, "predicted_state_idx_4"] = int(t5[3])
        template.loc[r, "predicted_state_idx_5"] = int(t5[4])
        template.loc[r, "predicted_latitude"] = float(ll[0])
        template.loc[r, "predicted_longitude"] = float(ll[1])

    out_path = Path(args.out_csv)
    template.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path}")


if __name__ == "__main__":
    main()
