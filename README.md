# GeoGuessr USA Street View (Panoramic) Challenge

Predict **where in the United States** a street view panorama was taken using **four directional images** captured at the same location.

This is a **dual-task** problem:

1. **State Classification** (33 states; indexed in `0–49`, non-consecutive)
2. **GPS Regression** (latitude + longitude)

Final submissions are scored using a weighted combination of **Top-K state classification** and **normalized Haversine distance** for GPS accuracy.

---

## Task Summary

Given a sample with four 256×256 RGB images:

- `North (0°)` → `image_north`
- `East (90°)` → `image_east`
- `South (180°)` → `image_south`
- `West (270°)` → `image_west`

You must predict:

- **Up to 5 ranked state indices** (`predicted_state_idx_1` ... `predicted_state_idx_5`)
- **Latitude** (`predicted_latitude`)
- **Longitude** (`predicted_longitude`)

---

## Dataset Overview

### Data Scale

- **Training samples:** 65,980
  - Total images: 263,920 (4 per sample)
- **Test samples:** 16,495
  - Total images: 65,980 (4 per sample)
- **States included:** 33 unique states
  - Encoded using `state_idx` values in the range `[0, 49]`
  - Indices are **not consecutive** (some numbers are missing)

### Image Format

- Resolution: **256×256**
- Color: **RGB**
- Format: **JPG**
- 4-direction coverage per sample: **North / East / South / West**

### File Structure

- `train_images/` — directory of training images
- `test_images/` — directory of test images
- `train_ground_truth.csv` — labels for training set
- `sample_submission.csv` — submission template
- `state_mapping.csv` — mapping between `state_idx` and state name

---

## CSV Schemas

### `train_ground_truth.csv`

Each row represents a location (sample) and references 4 images.

Columns:

- `sample_id`
- `image_north`, `image_east`, `image_south`, `image_west`
- `state` (string name)
- `state_idx` (integer, in `[0, 49]`, non-consecutive)
- `latitude` (float)
- `longitude` (float)

Example:

```csv
sample_id,image_north,image_east,image_south,image_west,state,state_idx,latitude,longitude
0,img_000000_north.jpg,img_000000_east.jpg,img_000000_south.jpg,img_000000_west.jpg,Maine,18,43.472421,-70.719764
1,img_000001_north.jpg,img_000001_east.jpg,img_000001_south.jpg,img_000001_west.jpg,Kentucky,16,37.138246,-83.370471
```

### `sample_submission.csv`

Template for test predictions.

Required outputs:

- `predicted_state_idx_1`
- `predicted_latitude`
- `predicted_longitude`

Optional outputs (for partial credit in state classification):

- `predicted_state_idx_2` ... `predicted_state_idx_5`

Example (single guess per sample):

```csv
sample_id,image_north,image_east,image_south,image_west,predicted_state_idx_1,predicted_state_idx_2,predicted_state_idx_3,predicted_state_idx_4,predicted_state_idx_5,predicted_latitude,predicted_longitude
0,img_000000_north.jpg,img_000000_east.jpg,img_000000_south.jpg,img_000000_west.jpg,4,-1,-1,-1,-1,37.7749,-122.4194
```

---

## Evaluation Metric

Overall score:

[
\text{Final Score} = 0.70 \times \text{Classification Score} + 0.30 \times \text{GPS Score}
]

### 1) Weighted Top-K State Classification (70%)

You may provide **up to 5 ranked** state predictions. Only the **first correct match** counts.

| Position | Weight |
| -------: | :----: |
|        1 |  1.00  |
|        2 |  0.60  |
|        3 |  0.40  |
|        4 |  0.25  |
|        5 |  0.15  |

Rules:

- Predictions are checked in order (1 → 5)
- Score is the weight of the **first position** that matches ground truth
- If no match in top-5: score = 0
- Duplicate state predictions are ignored (only first occurrence matters)

### 2) GPS Regression via Normalized Haversine (30%)

Compute mean great-circle distance error (km) across the test set.

[
\text{gps_score} = \max\left(0, 1 - \frac{\text{mean_distance_km}}{5000}\right)
]

Interpretation:

- Perfect (0 km): GPS score = 1.0
- ≥ 5000 km error: GPS score = 0.0

---

## Submission Requirements

### Required Columns

Your submission CSV must contain these columns (any order is acceptable):

- `sample_id`
- `image_north`, `image_east`, `image_south`, `image_west` (reference-only, not scored)
- `predicted_state_idx_1` (REQUIRED)
- `predicted_state_idx_2` (OPTIONAL)
- `predicted_state_idx_3` (OPTIONAL)
- `predicted_state_idx_4` (OPTIONAL)
- `predicted_state_idx_5` (OPTIONAL)
- `predicted_latitude` (REQUIRED)
- `predicted_longitude` (REQUIRED)

### Validation Rules

- Row count must match test set: **16,495 rows**
- `predicted_state_idx_1` must be in `[0, 49]`
- `predicted_latitude` must be in `[-90, 90]`
- `predicted_longitude` must be in `[-180, 180]`
- `predicted_state_idx_2` ... `predicted_state_idx_5` can be:

  - a valid state index in `[0, 49]`, or
  - `-1`, or
  - empty/NaN

---

## Notes on State Indices

- Valid prediction range is **0–49**
- Only **33 states** are present in the dataset
- Indices are **not consecutive** (e.g., index `2` may not exist)
- Use `state_mapping.csv` to map indices to state names

---

## Leaderboard Split

- **Public leaderboard:** 50% of test set (visible during competition)
- **Private leaderboard:** 50% of test set (final ranking)

This split reduces overfitting to the public test portion.

---

## Implementation Tips (Non-Prescriptive)

- Treat as **multi-view learning**: fuse four directions (early/late fusion)
- Consider multi-task learning with a shared backbone and two heads:

  - classification head for state
  - regression head for latitude/longitude

- Use top-5 output for better classification score (especially if uncertain)

---

## Quick FAQ

**Do I have to fill all 5 state predictions?**
No. Only `predicted_state_idx_1` is required. Use `-1` (or empty) for the rest.

**If I predict the correct GPS but miss the state, do I still get points?**
Yes. GPS contributes 30% of the final score.

**Are all US longitudes negative?**
For US locations, typically yes (West longitudes are negative), but your validation range is global.

---

## Training & Inference Workflow (Homework Submission)

Our final workflow is a **two-model pipeline**:

1. **GPS regression (latitude/longitude)** via a **DINOv3** backbone trained in this repo: `src/train.py` → `src/inference.py`
2. **State top-5 recommendations** via **OSV5M baseline** (`osv5m/baseline`): https://huggingface.co/osv5m/baseline

The final Kaggle submission is produced by **merging**:
- `predicted_latitude`, `predicted_longitude` from the DINOv3 model
- `predicted_state_idx_1..5` from the OSV5M state recommender

### 0) Environment / Data

- Create env + install deps: `pip install -r requirements.txt`
- Put the dataset under `data/` (default layout described above).
- First run will download Hugging Face weights for DINOv3 / OSV5M (unless already cached).

### 1) Train DINOv3 for GPS (this repo)

We train a 4-view fusion model with a DINOv3 encoder (frozen backbone + LoRA adapters by default) and a GPS head.

```bash
.venv/bin/python src/train.py \
  --encoder_type dinov3 \
  --hf_model_id facebook/dinov3-vit7b16-pretrain-lvd1689m \
  --data_dir data \
  --out_dir checkpoints_dinov3 \
  --epochs 5 \
  --batch_size 32 \
  --gradient_accumulation_steps 4 \
  --val_split 0.05
```

Multi-GPU (DDP) example (2 GPUs). `--batch_size` is per-GPU:

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 src/train.py \
  --encoder_type dinov3 \
  --hf_model_id facebook/dinov3-vit7b16-pretrain-lvd1689m \
  --data_dir data \
  --out_dir checkpoints_dinov3_ddp \
  --epochs 5 \
  --batch_size 32 \
  --gradient_accumulation_steps 4 \
  --val_split 0.05
```

Slurm example (`sbatch --gres=gpu:2`): run a single multi-process job that uses both GPUs:

```bash
srun --ntasks=1 --cpus-per-task=12 --gres=gpu:2 \
  .venv/bin/torchrun --standalone --nproc_per_node=2 src/train.py \
    --encoder_type dinov3 \
    --hf_model_id facebook/dinov3-vit7b16-pretrain-lvd1689m \
    --data_dir data \
    --out_dir checkpoints_dinov3_ddp \
    --epochs 5 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --val_split 0.05
```

Outputs:
- `checkpoints_dinov3/model_epoch_*.pt` (per-epoch checkpoints)
- `checkpoints_dinov3/model_best.pt` (best checkpoint by validation metric)

### 2) Run DINOv3 inference (this repo)

This writes a full submission CSV. In our final submission, we **keep only the lat/lon columns** from this file.

```bash
.venv/bin/python src/inference.py \
  --ckpt checkpoints_dinov3/model_best.pt \
  --data_dir data \
  --out_csv submission_dinov3.csv
```

Multi-GPU inference (DDP) example (2 GPUs):

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 src/inference.py \
  --ckpt checkpoints_dinov3/model_best.pt \
  --data_dir data \
  --out_csv submission_dinov3.csv
```

Slurm example (2 GPUs):

```bash
srun --ntasks=1 --cpus-per-task=12 --gres=gpu:2 \
  .venv/bin/torchrun --standalone --nproc_per_node=2 src/inference.py \
    --ckpt checkpoints_dinov3/model_best.pt \
    --data_dir data \
    --out_csv submission_dinov3.csv
```

### 3) Train/Run OSV5M for state top-5 (external; runs inside osv5m repo)

We use the pretrained OSV5M geolocalization backbone (`osv5m/baseline`) and train a lightweight state classifier head on top.
The code we used for this finetune is a standalone script that must be executed with the **OSV5M GitHub repo on `PYTHONPATH`**
(it imports `from models.huggingface import Geolocalizer` from that repo).

High-level steps performed by the script:
- Read `train_ground_truth.csv`, build a train/val split, and map sparse `state_idx` → contiguous class IDs
- For each sample, extract OSV5M CLS embeddings for the 4 views, mean-pool, and train a linear head (optionally unfreezing the backbone later)
- Select `best.pt` by validation top-1; then run inference on `sample_submission.csv` to write top-5 `state_idx` columns

Example (train + write a state-only submission):
```bash
python osv5m_state_finetune_script.py \
  --data_dir /path/to/geo-guesser/data \
  --state_mapping /path/to/geo-guesser/data/state_mapping.csv \
  --out_dir checkpoints_state \
  --epochs 3 \
  --make_submission \
  --submission_csv submission_osv5m.csv \
  --fill_latlon nan
```

Example (inference-only from a checkpoint):
```bash
python osv5m_state_finetune_script.py \
  --data_dir /path/to/geo-guesser/data \
  --state_mapping /path/to/geo-guesser/data/state_mapping.csv \
  --out_dir checkpoints_state \
  --inference_only \
  --ckpt_path checkpoints_state/best.pt \
  --submission_csv submission_osv5m.csv \
  --fill_latlon nan
```

### 4) Merge OSV5M states + DINOv3 GPS into the final submission

Both CSVs contain `sample_id`, so we merge by `sample_id` and copy over the columns we want:

```bash
python - <<'PY'
import pandas as pd

gps = pd.read_csv("submission_dinov3.csv")
states = pd.read_csv("submission_osv5m.csv").set_index("sample_id")

out = gps.copy()
for c in [f"predicted_state_idx_{i}" for i in range(1, 6)]:
    out[c] = states.loc[gps["sample_id"], c].values

out.to_csv("submission_final.csv", index=False)
print("Wrote submission_final.csv")
PY
```

### Optional: StreetCLIP end-to-end (single model baseline)

This repo also supports fine-tuning StreetCLIP (`geolocal/StreetCLIP`) as the encoder (4-view fusion + state classification + GPS regression).

```bash
.venv/bin/python src/train.py \
  --encoder_type streetclip \
  --hf_model_id geolocal/StreetCLIP \
  --data_dir data \
  --out_dir checkpoints_streetclip \
  --epochs 5 \
  --batch_size 32 \
  --gradient_accumulation_steps 4 \
  --val_split 0.05
```

Multi-GPU (DDP) example (2 GPUs). `--batch_size` is per-GPU:

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 src/train.py \
  --encoder_type streetclip \
  --hf_model_id geolocal/StreetCLIP \
  --data_dir data \
  --out_dir checkpoints_streetclip_ddp \
  --epochs 5 \
  --batch_size 32 \
  --gradient_accumulation_steps 4 \
  --val_split 0.05
```

Generate a submission from a checkpoint:

```bash
.venv/bin/python src/inference.py \
  --ckpt checkpoints_streetclip/model_best.pt \
  --data_dir data \
  --out_csv submission_streetclip.csv
```

Multi-GPU inference (DDP) example (2 GPUs):

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 src/inference.py \
  --ckpt checkpoints_streetclip/model_best.pt \
  --data_dir data \
  --out_csv submission_streetclip.csv
```
