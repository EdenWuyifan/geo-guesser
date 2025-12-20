from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# osv5m repo must be available on PYTHONPATH
from models.huggingface import Geolocalizer


# ----------------------------
# Repro / helpers
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def topk_indices(logits: torch.Tensor, k: int = 5) -> torch.Tensor:
    return torch.topk(logits, k=k, dim=-1).indices  # (B,k)


def safe_torch_load(path: Path, map_location="cpu"):
    return torch.load(str(path), map_location=map_location)


def resolve_path(data_dir: Path, path_or_name: str) -> Path:
    """
    Prefer `data_dir / path_or_name`, but fall back to `path_or_name` if it exists as-is.
    Keeps old defaults working while supporting alternate layouts.
    """
    candidate = data_dir / path_or_name
    if candidate.exists():
        return candidate
    raw = Path(path_or_name)
    if raw.exists():
        return raw
    return candidate


def _strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix) :] if s.startswith(prefix) else s


def sanitize_state_dict_keys(state_dict: Dict) -> Dict:
    """
    Best-effort compatibility with checkpoints saved in different wrappers.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    out = {}
    for k, v in state_dict.items():
        k = _strip_prefix(str(k), "module.")
        k = _strip_prefix(k, "model.")
        out[k] = v
    return out


def unpack_checkpoint(ckpt_obj) -> Tuple[Dict, Dict[int, int] | None, Dict]:
    """
    Returns: (state_dict, contig_to_state_idx, meta)
    Supports:
      - our saved format: {"model": ..., "contig_to_state_idx": ..., "meta": ...}
      - lightning-ish: {"state_dict": ...}
      - raw state_dict
    """
    contig_to_state_idx = None
    meta: Dict = {}

    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj:
        state_dict = ckpt_obj["model"]
        contig_to_state_idx = ckpt_obj.get("contig_to_state_idx", None)
        meta = ckpt_obj.get("meta", {}) or {}
        return sanitize_state_dict_keys(state_dict), contig_to_state_idx, meta

    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return sanitize_state_dict_keys(ckpt_obj["state_dict"]), None, {}

    if isinstance(ckpt_obj, dict):
        return sanitize_state_dict_keys(ckpt_obj), None, {}

    raise TypeError(f"Unsupported checkpoint object type: {type(ckpt_obj)}")


def infer_num_classes_from_state_dict(state_dict: Dict) -> int | None:
    if not isinstance(state_dict, dict):
        return None
    for key in ("head.weight", "classifier.weight", "fc.weight"):
        w = state_dict.get(key, None)
        if hasattr(w, "shape") and len(w.shape) == 2:
            return int(w.shape[0])
    return None


# ----------------------------
# Dataset
# ----------------------------
class StreetViewStateDataset(Dataset):
    """
    Returns dict:
      images: (4,3,H,W) float
      y: contiguous class [0..C-1] if train/val
      sample_id: int (always)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        transform,
        is_train: bool,
        state_idx_to_contig: Dict[int, int] | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_train = is_train
        self.state_idx_to_contig = state_idx_to_contig
        self.dirs = ["image_north", "image_east", "image_south", "image_west"]

        if self.is_train and self.state_idx_to_contig is None:
            raise ValueError("state_idx_to_contig is required for training/validation.")

    def __len__(self) -> int:
        return len(self.df)

    def _load_one(self, fname: str) -> torch.Tensor:
        p = self.img_dir / fname
        img = Image.open(p).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        imgs = [self._load_one(row[c]) for c in self.dirs]
        imgs = torch.stack(imgs, dim=0)  # (4,3,H,W)

        out = {
            "images": imgs,
            "sample_id": torch.tensor(int(row["sample_id"]), dtype=torch.long),
        }

        if self.is_train:
            state_idx = int(row["state_idx"])
            y = self.state_idx_to_contig[state_idx]
            out["y"] = torch.tensor(y, dtype=torch.long)

        return out


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([b["images"] for b in batch], dim=0)  # (B,4,3,H,W)
    sample_id = torch.stack([b["sample_id"] for b in batch], dim=0)  # (B,)
    out = {"images": images, "sample_id": sample_id}
    if "y" in batch[0]:
        y = torch.stack([b["y"] for b in batch], dim=0)
        out["y"] = y
    return out


# ----------------------------
# Model
# ----------------------------
class OSV5MStateRecommender(nn.Module):
    """
    Extract CLS embeddings per view from osv5m backbone, mean-pool across 4 views,
    then linear head for state logits.
    """

    def __init__(self, geoloc: Geolocalizer, num_classes: int) -> None:
        super().__init__()
        self.geoloc = geoloc

        # Freeze everything by default (we may unfreeze later)
        for p in self.geoloc.parameters():
            p.requires_grad = False

        self.embed_dim = 1024  # default guess for osv5m/baseline
        self.head = nn.Linear(self.embed_dim, num_classes)

    @torch.no_grad()
    def _infer_dim(self, device: torch.device) -> int:
        x = torch.zeros(1, 3, 224, 224, device=device)
        out = self.geoloc.backbone({"img": x})
        return int(out.shape[-1])

    def maybe_init_dim(self, device: torch.device) -> None:
        try:
            d = self._infer_dim(device)
            if d != self.embed_dim:
                self.embed_dim = d
                self.head = nn.Linear(self.embed_dim, self.head.out_features).to(device)
        except Exception:
            pass

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.geoloc.parameters():
            p.requires_grad = trainable

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B,4,3,H,W)
        returns logits: (B,C)
        """
        b, v, c, h, w = images.shape
        x = images.view(b * v, c, h, w)

        # backbone_out: (B*V, L, D); take CLS token
        backbone_out = self.geoloc.backbone({"img": x})
        feats = backbone_out[:, 0, :]  # (B*V, D)
        feats = feats.view(b, v, -1).mean(dim=1)  # (B, D)

        return self.head(feats)


# ----------------------------
# Train / eval
# ----------------------------
@torch.no_grad()
def eval_topk(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        logits = model(images)
        total += y.size(0)

        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == y).sum().item()

        top5 = logits.topk(5, dim=1).indices
        correct5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()

    return 100.0 * correct1 / total, 100.0 * correct5 / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = ce(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        bs = y.size(0)
        total += bs
        total_loss += loss.detach().float().item() * bs

    return total_loss / max(total, 1)


# ----------------------------
# Checkpointing
# ----------------------------
@dataclass
class BestTracker:
    best_top1: float = -1.0
    best_path: Path | None = None

    def consider(self, top1: float, ckpt_path: Path) -> bool:
        if top1 > self.best_top1:
            self.best_top1 = top1
            self.best_path = ckpt_path
            return True
        return False


def save_checkpoint(
    path: Path,
    epoch: int,
    model: OSV5MStateRecommender,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    contig_to_state_idx: Dict[int, int],
    meta: Dict,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "contig_to_state_idx": contig_to_state_idx,
        "meta": meta,
    }
    torch.save(payload, str(path))


def find_best_checkpoint(out_dir: Path) -> Path | None:
    best = out_dir / "best.pt"
    if best.exists():
        return best

    # fallback: choose the checkpoint with the highest recorded val_top1 in filename json
    candidates = sorted(out_dir.glob("epoch_*.pt"))
    if not candidates:
        return None

    # choose latest if no metrics
    return candidates[-1]


# ----------------------------
# Submission
# ----------------------------
@torch.no_grad()
def write_submission_top5(
    model: nn.Module,
    loader: DataLoader,
    template_csv: Path,
    contig_to_state_idx: Dict[int, int],
    out_csv: Path,
    device: torch.device,
    fill_latlon: str = "nan",  # "nan" | "zero" | "mean"
    train_mean_latlon: Tuple[float, float] | None = None,
) -> None:
    model.eval()
    template = pd.read_csv(template_csv)

    # Ensure columns exist
    for i in range(1, 6):
        col = f"predicted_state_idx_{i}"
        if col not in template.columns:
            template[col] = -1

    if "predicted_latitude" not in template.columns:
        template["predicted_latitude"] = np.nan
    if "predicted_longitude" not in template.columns:
        template["predicted_longitude"] = np.nan

    all_ids: List[int] = []
    all_top5: List[List[int]] = []

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        sample_id = batch["sample_id"].cpu().numpy().tolist()

        logits = model(images)
        top5_contig = topk_indices(logits, k=5).cpu().numpy()  # (B,5)

        for sid, row in zip(sample_id, top5_contig):
            mapped = [int(contig_to_state_idx[int(x)]) for x in row]
            all_ids.append(int(sid))
            all_top5.append(mapped)

    id_to_row = {int(sid): i for i, sid in enumerate(template["sample_id"].values)}
    for sid, top5 in zip(all_ids, all_top5):
        r = id_to_row[sid]
        for j in range(5):
            template.loc[r, f"predicted_state_idx_{j+1}"] = int(top5[j])

    # Lat/lon placeholders
    if fill_latlon == "zero":
        template["predicted_latitude"] = 0.0
        template["predicted_longitude"] = 0.0
    elif fill_latlon == "mean":
        if train_mean_latlon is None:
            raise ValueError(
                "train_mean_latlon must be provided when fill_latlon='mean'"
            )
        template["predicted_latitude"] = float(train_mean_latlon[0])
        template["predicted_longitude"] = float(train_mean_latlon[1])
    else:
        # keep NaN
        pass

    template.to_csv(out_csv, index=False)
    print(f"Wrote submission: {out_csv}")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--train_csv", type=str, default="train_ground_truth.csv")
    ap.add_argument("--test_csv", type=str, default="sample_submission.csv")
    ap.add_argument("--train_images", type=str, default="train_images")
    ap.add_argument("--test_images", type=str, default="test_images")
    ap.add_argument("--state_mapping", type=str, default="data/state_mapping.csv")

    ap.add_argument("--osv_model_id", type=str, default="osv5m/baseline")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument(
        "--finetune_after_epochs",
        type=int,
        default=-1,
        help="If >=0, unfreeze backbone after this many epochs and fine-tune for remaining epochs.",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--out_dir", type=str, default="checkpoints_state")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Checkpoint path for inference/resume (default: auto-pick best.pt in out_dir).",
    )
    ap.add_argument(
        "--inference_only",
        action="store_true",
        help="Skip training and run inference on test set using a checkpoint.",
    )

    ap.add_argument("--make_submission", action="store_true")
    ap.add_argument("--submission_csv", type=str, default="submission.csv")
    ap.add_argument(
        "--fill_latlon", type=str, default="nan", choices=["nan", "zero", "mean"]
    )
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_csv = resolve_path(data_dir, args.train_csv)
    test_csv = resolve_path(data_dir, args.test_csv)
    train_img_dir = resolve_path(data_dir, args.train_images)
    test_img_dir = resolve_path(data_dir, args.test_images)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build contiguous mapping from state_mapping.csv
    smap = pd.read_csv(args.state_mapping)
    valid_state_idxs = sorted(smap["state_idx"].unique().tolist())
    state_idx_to_contig = {orig: i for i, orig in enumerate(valid_state_idxs)}
    contig_to_state_idx = {v: k for k, v in state_idx_to_contig.items()}
    num_classes = len(valid_state_idxs)
    print(f"Using {num_classes} classes")

    want_submission = bool(args.make_submission or args.inference_only)
    do_train = (not args.inference_only) and (args.epochs > 0)

    train_mean_latlon: Tuple[float, float] | None = None
    if args.fill_latlon == "mean" and want_submission:
        # Only needed for filling lat/lon columns; skip if user selects nan/zero.
        df_for_mean = pd.read_csv(train_csv)
        train_mean_latlon = (
            float(df_for_mean["latitude"].mean()),
            float(df_for_mean["longitude"].mean()),
        )

    # --- Read train/val split (training only)
    if do_train:
        df = pd.read_csv(train_csv)
        df["state_contig"] = df["state_idx"].map(state_idx_to_contig)

        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=args.seed
        )
        tr_idx, va_idx = next(splitter.split(df, df["state_contig"].values))
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

    # --- CLIP normalization (OSV / CLIP)
    try:
        from torchvision.transforms import InterpolationMode

        bicubic = InterpolationMode.BICUBIC
    except Exception:
        bicubic = 3

    transform_fn = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=bicubic),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    train_loader = None
    val_loader = None
    if do_train:
        train_ds = StreetViewStateDataset(
            df_tr, train_img_dir, transform_fn, True, state_idx_to_contig
        )
        val_ds = StreetViewStateDataset(
            df_va, train_img_dir, transform_fn, True, state_idx_to_contig
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=4 if args.num_workers > 0 else None,
            collate_fn=collate,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=4 if args.num_workers > 0 else None,
            collate_fn=collate,
        )

    # --- Pick checkpoint path (for inference/resume)
    ckpt_path: Path | None = None
    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
    else:
        ckpt_path = find_best_checkpoint(out_dir)

    # --- If inference_only: read ckpt first to infer num_classes/mapping
    contig_to_state_idx_ckpt: Dict[int, int] | None = None
    if args.inference_only:
        if ckpt_path is None or (not ckpt_path.exists()):
            raise FileNotFoundError(
                f"Checkpoint not found. Provide --ckpt_path or ensure {out_dir / 'best.pt'} exists."
            )
        ckpt_obj = safe_torch_load(ckpt_path, map_location="cpu")
        state_dict, contig_to_state_idx_ckpt, _meta = unpack_checkpoint(ckpt_obj)
        inferred = None
        if contig_to_state_idx_ckpt is not None:
            inferred = len(contig_to_state_idx_ckpt)
        if inferred is None:
            inferred = infer_num_classes_from_state_dict(state_dict)
        if inferred is not None and inferred != num_classes:
            print(
                f"[inference_only] Overriding num_classes: {num_classes} -> {inferred} (from checkpoint)"
            )
            num_classes = int(inferred)

    # --- Load OSV5M geolocalizer and model
    geoloc = Geolocalizer.from_pretrained(args.osv_model_id).to(device)
    model = OSV5MStateRecommender(geoloc, num_classes=num_classes).to(device)
    model.maybe_init_dim(device)

    # param groups: head always, backbone optionally
    def make_optimizer(finetune: bool) -> torch.optim.Optimizer:
        params = [{"params": model.head.parameters(), "lr": args.lr_head}]
        if finetune:
            params.append({"params": model.geoloc.parameters(), "lr": args.lr_backbone})
        return torch.optim.AdamW(params, weight_decay=args.weight_decay)

    finetune_enabled = False
    model.set_backbone_trainable(False)
    optimizer = make_optimizer(finetune=False)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    tracker = BestTracker()

    # --- Resume support (optional)
    if args.resume:
        if ckpt_path is not None and ckpt_path.exists():
            ckpt_obj = safe_torch_load(ckpt_path, map_location="cpu")
            state_dict, _contig_to_state_idx, meta = unpack_checkpoint(ckpt_obj)
            model.load_state_dict(state_dict, strict=True)
            try:
                if isinstance(ckpt_obj, dict):
                    optimizer.load_state_dict(ckpt_obj.get("optimizer", {}))
                    scaler.load_state_dict(ckpt_obj.get("scaler", {}))
            except Exception:
                pass
            if isinstance(ckpt_obj, dict):
                start_epoch = int(ckpt_obj.get("epoch", 0)) + 1
            print(f"Resumed from {ckpt_path} (next epoch: {start_epoch})")
            # tracker from meta if present
            tracker.best_top1 = float(meta.get("best_top1", -1.0))
            tracker.best_path = ckpt_path

    # --- Train
    if do_train:
        assert train_loader is not None
        assert val_loader is not None

        for epoch in range(start_epoch, args.epochs + 1):
            # enable finetune after N epochs
            if (
                args.finetune_after_epochs >= 0
                and (epoch > args.finetune_after_epochs)
                and (not finetune_enabled)
            ):
                finetune_enabled = True
                model.set_backbone_trainable(True)
                optimizer = make_optimizer(finetune=True)
                print(
                    f"[Epoch {epoch}] Enabled finetuning: backbone unfrozen (lr_backbone={args.lr_backbone})"
                )

            tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
            top1, top5 = eval_topk(model, val_loader, device)
            print(
                f"Epoch {epoch}/{args.epochs} | train_loss={tr_loss:.4f} | val_top1={top1:.2f}% | val_top5={top5:.2f}%"
            )

            # save epoch ckpt
            epoch_path = out_dir / f"epoch_{epoch:03d}.pt"
            meta = {
                "val_top1": top1,
                "val_top5": top5,
                "train_loss": tr_loss,
                "best_top1": max(tracker.best_top1, top1),
                "finetune_enabled": finetune_enabled,
                "args": vars(args),
            }
            save_checkpoint(
                epoch_path, epoch, model, optimizer, scaler, contig_to_state_idx, meta
            )

            # update best
            if tracker.consider(top1, epoch_path):
                best_path = out_dir / "best.pt"
                save_checkpoint(
                    best_path,
                    epoch,
                    model,
                    optimizer,
                    scaler,
                    contig_to_state_idx,
                    meta,
                )
                print(f"  Saved best: {best_path} (val_top1={top1:.2f}%)")

    # --- Submission
    if want_submission:
        if ckpt_path is None or (not ckpt_path.exists()):
            raise FileNotFoundError(
                f"No checkpoint found. Provide --ckpt_path or ensure {out_dir / 'best.pt'} exists."
            )

        ckpt_obj = safe_torch_load(ckpt_path, map_location="cpu")
        state_dict, contig_to_state_idx_ckpt2, _meta = unpack_checkpoint(ckpt_obj)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(
                f"[warn] Strict checkpoint load failed ({e}); retrying with strict=False"
            )
            model.load_state_dict(state_dict, strict=False)

        # ensure mapping is consistent with checkpoint if it contains one
        contig_to_state_idx_ckpt = (
            contig_to_state_idx_ckpt or contig_to_state_idx_ckpt2 or contig_to_state_idx
        )

        test_df = pd.read_csv(test_csv)
        test_ds = StreetViewStateDataset(test_df, test_img_dir, transform_fn, False)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=4 if args.num_workers > 0 else None,
            collate_fn=collate,
        )

        out_csv = Path(args.submission_csv)
        write_submission_top5(
            model=model,
            loader=test_loader,
            template_csv=test_csv,
            contig_to_state_idx=contig_to_state_idx_ckpt,
            out_csv=out_csv,
            device=device,
            fill_latlon=args.fill_latlon,
            train_mean_latlon=train_mean_latlon,
        )


if __name__ == "__main__":
    main()
