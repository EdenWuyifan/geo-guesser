from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class GeoCells:
    centroids_latlon: np.ndarray  # (C,2) float32 [lat, lon]
    cell_ids: np.ndarray  # (N,) int64    index in [0..C-1]
    meta: dict  # extra info (e.g., s2_level, cell_tokens)


def _latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _xyz_to_latlon(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    lat = np.arcsin(xyz[:, 2])
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.stack([np.rad2deg(lat), np.rad2deg(lon)], axis=1).astype(np.float32)


def build_geo_cells(
    train_csv: Path,
    method: Literal["s2", "kmeans3d"] = "s2",
    # S2 parameters
    s2_level: int = 6,
    # KMeans(3D) parameters
    num_cells: int = 1024,
    random_state: int = 42,
) -> GeoCells:
    """
    Build geographic cells for geolocation training.

    method="s2":
      - deterministic binning via S2 cell IDs at a given level
      - number of unique cells is data-dependent
      - set s2_level higher -> smaller cells (more unique)

    method="kmeans3d":
      - data-adaptive clustering on 3D unit sphere
      - number of cells is exactly num_cells

    Returns:
      GeoCells with:
        - centroids_latlon: (C,2)
        - cell_ids: (N,)
        - meta: useful diagnostics (and S2 cell tokens when method="s2")
    """
    import pandas as pd

    df = pd.read_csv(train_csv)
    lat = df["latitude"].to_numpy(dtype=np.float64)
    lon = df["longitude"].to_numpy(dtype=np.float64)

    if method == "kmeans3d":
        from sklearn.cluster import KMeans

        xyz = _latlon_to_xyz(lat, lon)  # (N,3)
        km = KMeans(
            n_clusters=num_cells,
            random_state=random_state,
            n_init="auto",
            verbose=1,
        )
        cell_ids = km.fit_predict(xyz).astype(np.int64)  # (N,)
        centers_xyz = km.cluster_centers_.astype(np.float32)  # (C,3)
        centroids_latlon = _xyz_to_latlon(centers_xyz)  # (C,2)

        return GeoCells(
            centroids_latlon=centroids_latlon,
            cell_ids=cell_ids,
            meta={"method": "kmeans3d", "num_cells": int(num_cells)},
        )

    if method == "s2":
        # Requires: pip install s2sphere
        try:
            from s2sphere import CellId, LatLng
        except Exception as e:
            raise ImportError(
                "S2 requested but s2sphere is not installed. "
                "Install with: pip install s2sphere  (or switch method='kmeans3d')."
            ) from e

        # Map each sample to an S2 cell token at the desired level
        tokens = []
        for la, lo in zip(lat, lon):
            ll = LatLng.from_degrees(float(la), float(lo))
            cid = CellId.from_lat_lng(ll).parent(s2_level)
            tokens.append(cid.to_token())

        tokens = np.array(tokens, dtype=object)  # (N,)

        # Factorize unique tokens -> contiguous [0..C-1]
        unique_tokens, inv = np.unique(tokens, return_inverse=True)
        cell_ids = inv.astype(np.int64)  # (N,)

        # Compute centroids for each cell via the S2 cell center
        centroids_latlon = np.zeros((len(unique_tokens), 2), dtype=np.float32)
        for i, tok in enumerate(unique_tokens):
            cid = CellId.from_token(str(tok))
            ll = cid.to_lat_lng()
            centroids_latlon[i, 0] = ll.lat().degrees
            centroids_latlon[i, 1] = ll.lng().degrees

        return GeoCells(
            centroids_latlon=centroids_latlon,
            cell_ids=cell_ids,
            meta={
                "method": "s2",
                "s2_level": int(s2_level),
                "num_cells": int(len(unique_tokens)),
                "cell_tokens": unique_tokens.tolist(),  # useful for debugging/repro
            },
        )

    raise ValueError(f"Unknown method: {method}")
