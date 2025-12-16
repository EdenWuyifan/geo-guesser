# GeoGuessr USA — State-of-the-Art Training Plan (DINOv3 + Multi-View Fusion)

This document describes a **high-performance, research-grade plan** for solving the GeoGuessr USA Kaggle challenge using **per-view encoding with learned fusion**, a **DINOv3 backbone**, and a **cell-based geolocation head** optimized for the competition’s **weighted top-K classification + GPS regression** metric.

The plan is written to be directly actionable in a GitHub repository and can be implemented incrementally.

---

## 1. High-Level System Overview

**Input (per sample):**

- 4 RGB images (256×256): North, East, South, West

**Backbone:**

- DINOv3 Vision Transformer (frozen → partially fine-tuned)

**Core Components:**

1. Per-view visual encoder (shared weights)
2. Direction-aware fusion transformer
3. Dual heads:
   - State classification (Top-5 optimized)
   - GPS prediction (cell classification + residual regression)

**Output:**

- Top-5 ranked `state_idx`
- Latitude, longitude

---

## 2. Data Pipeline

### 2.1 Dataset Layout

```

data/
├── train_images/
├── test_images/
├── train_ground_truth.csv
├── sample_submission.csv
├── state_mapping.csv

```

### 2.2 Dataset Object (Conceptual)

Each training item yields:

```python
{
  "images": {
    "north": Tensor[3, H, W],
    "east":  Tensor[3, H, W],
    "south": Tensor[3, H, W],
    "west":  Tensor[3, H, W],
  },
  "state_idx": int,
  "latitude": float,
  "longitude": float
}
```

### 2.3 Augmentation Policy

Safe augmentations for street-view imagery:

- Color jitter (brightness, contrast, saturation)
- Mild random resized crop
- Gaussian blur (low probability)

⚠️ **Do not** randomly flip unless direction labels are swapped consistently.

---

## 3. Per-View Encoding (Shared DINOv3)

### 3.1 Encoder Design

- One shared **DINOv3 ViT** processes each directional image independently.
- Output: one embedding vector per view.

Example:

```
north → DINOv3 → z_n ∈ ℝ^D
east  → DINOv3 → z_e ∈ ℝ^D
south → DINOv3 → z_s ∈ ℝ^D
west  → DINOv3 → z_w ∈ ℝ^D
```

### 3.2 Freezing Strategy

**Phase 1**

- Freeze entire DINOv3 backbone
- Train only fusion + heads

**Phase 2 (optional)**

- Unfreeze last N transformer blocks or apply LoRA
- Reduce learning rate by 5–10×

---

## 4. Direction-Aware Fusion Module

### 4.1 Direction Embeddings

Assign a learned embedding to each direction:

```
E_dir ∈ {E_N, E_E, E_S, E_W}
```

Each view token becomes:

```
t_i = z_i + E_dir(i)
```

### 4.2 Fusion Transformer

- Input tokens: `[t_N, t_E, t_S, t_W]`
- Architecture:

  - 2–4 transformer encoder layers
  - Multi-head self-attention

- Output:

  - Either mean-pooled token
  - Or a learned `[CLS]` token

Final fused representation:

```
z_fused ∈ ℝ^D
```

---

## 5. State Classification Head (Top-K Optimized)

### 5.1 Head Architecture

```
z_fused
  → LayerNorm
  → Linear(D → 33)
  → Softmax
```

(Only 33 valid states; map to/from `state_idx` via `state_mapping.csv`.)

### 5.2 Loss

- Cross-entropy with label smoothing (ε = 0.05–0.1)

### 5.3 Inference

- Always output **top-5 unique state indices**
- Sorted by probability
- Directly aligns with weighted Top-K metric

---

## 6. GPS Head: Cell Classification + Residual Regression

### 6.1 Geo-Cell Construction (Offline Step)

Choose one:

- **S2 cells** (recommended)
- **K-Means** over 3D Earth coordinates

Target:

- ~2,000–10,000 cells
- Each cell has a centroid `(lat_c, lon_c)`

### 6.2 Model Outputs

From `z_fused`:

1. `p(cell | image)` — softmax over geo-cells
2. `(Δlat, Δlon)` — residual offset

### 6.3 Coordinate Reconstruction

At inference:

```
(lat_pred, lon_pred) =
  centroid(top_cell) + residual
```

Optional:

- Probability-weighted centroid over top-K cells

---

## 7. Loss Function

Let:

- `L_state` = state cross-entropy
- `L_cell` = geo-cell cross-entropy
- `L_gps` = Huber or Haversine loss on final coordinates

Total loss:

```
L = λ_state · L_state
  + λ_cell · L_cell
  + λ_gps  · L_gps
```

Recommended starting weights:

```
λ_state = 1.0
λ_cell  = 1.0
λ_gps   = 1.0
```

Tune `λ_state` upward if Top-5 accuracy lags.

---

## 8. Training Schedule

### Phase 1 — Head Training

- Frozen DINOv3
- Train fusion + heads
- 5–15 epochs

### Phase 2 — Partial Fine-Tuning

- Unfreeze last backbone blocks or add LoRA
- Lower LR
- 5–10 epochs

### Optimizer

- AdamW
- Cosine LR schedule
- 5% warmup
- Mixed precision (AMP)

---

## 9. Inference Enhancements

### 9.1 Test-Time Augmentation (TTA)

- Multi-crop / resize
- Average logits across TTAs

### 9.2 Ensemble

- Average predictions from:

  - Multiple checkpoints
  - Different DINOv3 sizes (if available)

### 9.3 GPS Safety Net

To avoid catastrophic GPS penalties:

- If state confidence is low:

  - Predict centroid of predicted state (train prior)
  - Or weighted centroid of top geo-cells

---

## 10. Submission Mapping

| Output Field               | Source              |
| -------------------------- | ------------------- |
| `predicted_state_idx_1..5` | Top-5 state softmax |
| `predicted_latitude`       | GPS head output     |
| `predicted_longitude`      | GPS head output     |

Ensure:

- All required fields filled
- Latitude ∈ [-90, 90]
- Longitude ∈ [-180, 180]

---

## 11. Suggested Repository Structure

```
geoguessr-usa/
├── data/
├── src/
│   ├── datasets/
│   ├── models/
│   │   ├── dinov3_encoder.py
│   │   ├── fusion_transformer.py
│   │   ├── state_head.py
│   │   └── geo_head.py
│   ├── losses/
│   ├── train.py
│   └── inference.py
├── scripts/
│   ├── build_geo_cells.py
│   └── submit.py
├── configs/
└── README.md
```

---

## 12. Why This Works Well for This Competition

- Multi-view fusion exploits full 360° context
- DINOv3 provides strong, generalizable representations
- Cell-based geolocation stabilizes GPS regression
- Top-K-aware state prediction maximizes classification score
- Ensemble + TTA align with Kaggle leaderboard dynamics

---

If you want, I can next:

- Write a **minimal PyTorch model skeleton** matching this plan
- Design the **geo-cell builder script**
- Provide a **single-GPU training config** tailored to your class hardware
