from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


@dataclass(frozen=True)
class StateIndexMapper:
    """
    Maps between:
      - Kaggle/original state_idx (non-consecutive, 0..49 domain but only 33 present)
      - contiguous class_id in [0..num_states-1] for training
    """

    state_idx_to_class: Dict[int, int]
    class_to_state_idx: List[int]
    class_to_state_name: List[str]

    @property
    def num_states(self) -> int:
        return len(self.class_to_state_idx)

    @staticmethod
    def from_csv(path: str | Path) -> "StateIndexMapper":
        path = Path(path)
        df = pd.read_csv(path)
        if not {"state_idx", "state"}.issubset(df.columns):
            raise ValueError(f"{path} must contain columns: state_idx,state")

        # Keep deterministic ordering by ascending state_idx
        df = df.sort_values("state_idx").reset_index(drop=True)

        class_to_state_idx = [int(x) for x in df["state_idx"].tolist()]
        class_to_state_name = [str(x) for x in df["state"].tolist()]
        state_idx_to_class = {si: i for i, si in enumerate(class_to_state_idx)}

        return StateIndexMapper(
            state_idx_to_class=state_idx_to_class,
            class_to_state_idx=class_to_state_idx,
            class_to_state_name=class_to_state_name,
        )

    def to_class(self, state_idx: int) -> int:
        if state_idx not in self.state_idx_to_class:
            raise KeyError(f"state_idx={state_idx} not found in state_mapping.csv")
        return self.state_idx_to_class[state_idx]

    def to_state_idx(self, class_id: int) -> int:
        if class_id < 0 or class_id >= self.num_states:
            raise IndexError(
                f"class_id={class_id} out of range [0,{self.num_states-1}]"
            )
        return self.class_to_state_idx[class_id]


@dataclass(frozen=True)
class StreetViewSample:
    row_idx: int  # CSV row index for efficient cell_id lookup
    sample_id: int
    images: Dict[str, torch.Tensor]  # {"north": [3,H,W], ...}
    state_class: Optional[int]  # contiguous class id (0..32), None for test
    state_idx: Optional[int]  # original Kaggle index, None for test
    latitude: Optional[float]  # None for test
    longitude: Optional[float]  # None for test


def build_transform(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    is_train: bool,
) -> T.Compose:
    ops = [T.Resize((image_size, image_size))]
    if is_train:
        ops += [
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        ]
    ops += [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(ops)


class StreetViewDataset(Dataset):
    """
    Train CSV must include: state_idx, latitude, longitude.
    This dataset converts original 'state_idx' -> contiguous 'state_class' via StateIndexMapper.
    """

    def __init__(
        self,
        csv_path: str | Path,
        images_dir: str | Path,
        is_train: bool,
        state_mapper: Optional[StateIndexMapper] = None,
        image_size: int = 256,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.is_train = is_train
        self.state_mapper = state_mapper

        self.df = pd.read_csv(self.csv_path)

        required = {
            "sample_id",
            "image_north",
            "image_east",
            "image_south",
            "image_west",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {csv_path}: {sorted(missing)}"
            )

        if is_train:
            for col in ["state_idx", "latitude", "longitude"]:
                if col not in self.df.columns:
                    raise ValueError(f"Training CSV must include column: {col}")
            if self.state_mapper is None:
                raise ValueError(
                    "state_mapper is required for training to map state_idx -> state_class."
                )

        self.transform = build_transform(
            image_size=image_size,
            mean=mean,
            std=std,
            is_train=is_train,
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, filename: str) -> torch.Tensor:
        path = self.images_dir / filename
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)

    def __getitem__(self, idx: int) -> StreetViewSample:
        row = self.df.iloc[idx]
        sample_id = int(row["sample_id"])

        images = {
            "north": self._load_image(row["image_north"]),
            "east": self._load_image(row["image_east"]),
            "south": self._load_image(row["image_south"]),
            "west": self._load_image(row["image_west"]),
        }

        if not self.is_train:
            return StreetViewSample(
                row_idx=idx,
                sample_id=sample_id,
                images=images,
                state_class=None,
                state_idx=None,
                latitude=None,
                longitude=None,
            )

        state_idx = int(row["state_idx"])
        state_class = self.state_mapper.to_class(state_idx)

        return StreetViewSample(
            row_idx=idx,
            sample_id=sample_id,
            images=images,
            state_class=state_class,
            state_idx=state_idx,
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
        )


DIRECTIONS = ("north", "east", "south", "west")


def collate_streetview(batch: List[StreetViewSample]) -> Dict[str, torch.Tensor]:
    sample_ids = torch.tensor([s.sample_id for s in batch], dtype=torch.long)
    row_indices = torch.tensor([s.row_idx for s in batch], dtype=torch.long)

    views = []
    for s in batch:
        views.append(torch.stack([s.images[d] for d in DIRECTIONS], dim=0))  # (4,3,H,W)
    images = torch.stack(views, dim=0)  # (B,4,3,H,W)

    out: Dict[str, torch.Tensor] = {
        "sample_id": sample_ids,
        "row_idx": row_indices,
        "images": images,
    }

    if batch[0].state_class is not None:
        out["state_class"] = torch.tensor(
            [s.state_class for s in batch], dtype=torch.long
        )
        out["latlon"] = torch.tensor(
            [[s.latitude, s.longitude] for s in batch], dtype=torch.float32
        )

    return out
