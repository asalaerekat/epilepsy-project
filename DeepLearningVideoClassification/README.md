# DeepLearningVideoClassification

3D video classification pipeline for binary seizure-type prediction with:
- grouped K-fold cross-validation on internal data,
- final model training on all internal data,
- external test-set evaluation,
- 3D Grad-CAM visualization for trained models.

Main scripts:
- `train.py` for training/evaluation workflow.
- `scripts/make_gradcam_video.py` for Grad-CAM overlays and CAM tensor export.
- `make_gradcam_video.py` as a backward-compatible wrapper to the script above.

## What The Pipeline Does

In `--run_mode train`, `train.py` runs:
1. Grouped K-fold CV (grouped by patient ID from filename prefix).
2. Per-fold checkpoint and metrics export.
3. CV summary export (including stop epoch statistics).
4. Final threshold selection (`avg Youden` or `0.5`).
5. Final model training on all internal data using median fold stop epoch.
6. External evaluation of `final_model.pth`.

In `--run_mode test_only`, `train.py` loads a saved model and evaluates an external dataset.

## Data Requirements

Videos must end with one of:
- `PNEE.mp4`
- `FDS.mp4`
- `GTC.mp4`
- `FocalBilateralTC.mp4`
- `ES.mp4`

Label mapping:
- Class `0`: `PNEE`, `FDS`
- Class `1`: `GTC`, `FocalBilateralTC`, `ES`

Example filename:
- `109_1_FDS.mp4`

Patient grouping for CV uses the prefix before first underscore:
- `109_1_FDS.mp4` -> patient group `109`

## Main Arguments

- `--run_mode {train,test_only}`
- `--video_dir` (internal dataset path)
- `--external_video_dir` (external test/eval path)
- `--k_folds` (default `5`)
- `--num_workers` (train/CV DataLoader workers)
- `--eval_num_workers` (external eval DataLoader workers, default `0`)
- `--eval_batch_size` (external eval batch size, default `1`)
- `--epochs` (max epochs used in CV)
- `--use_avg_youden_threshold` (if set, final threshold = mean K-fold Youden; else `0.5`)
- `--checkpoint_path` (test-only checkpoint; default `<output_dir>/final_model.pth`)
- `--threshold` (test-only threshold override)

Model:
- `--model resnet3d` is implemented.
- `--model vmz` currently raises `NotImplementedError`.

## Usage

Run from this directory:

```bash
cd DeepLearningVideoClassification
```

Train full workflow:

```bash
python train.py \
  --run_mode train \
  --video_dir /path/to/internal_videos \
  --external_video_dir /path/to/external_videos \
  --model resnet3d \
  --pretrained \
  --k_folds 5 \
  --epochs 20 \
  --batch_size 4
```

Train full workflow and use average K-fold Youden threshold:

```bash
python train.py \
  --run_mode train \
  --video_dir /path/to/internal_videos \
  --external_video_dir /path/to/external_videos \
  --use_avg_youden_threshold
```

Test only (use saved final model and saved threshold if present):

```bash
python train.py \
  --run_mode test_only \
  --external_video_dir /path/to/external_videos \
  --eval_batch_size 1 \
  --eval_num_workers 0
```

Test only with explicit checkpoint and threshold:

```bash
python train.py \
  --run_mode test_only \
  --external_video_dir /path/to/external_videos \
  --checkpoint_path /path/to/final_model.pth \
  --threshold 0.5
```

## 3D Grad-CAM Visualization

`gradcam_3d.py` provides `GradCAM3D` for video models.

Behavior:
- Input tensor: `(N,C,T,H,W)` normalized clip.
- Output CAM: `(N,T,H,W)` in `[0,1]`.
- CAM is upsampled to input `(T,H,W)` with trilinear interpolation.
- Binary head `(N,1)` supports class-specific CAM with `--class_idx`:
  - `--class_idx 1`: seizure evidence (`+logit`)
  - `--class_idx 0`: non-seizure evidence (`-logit`)
  - omitted: defaults to seizure evidence (`class 1`)
- Multiclass `(N,K)` supports selected class index (`--class_idx`), or argmax when not set.
- Default target layer is `model.layer4[-1].conv2` (works with torchvision `r3d_18`).

Generate CAM overlays:

```bash
python scripts/make_gradcam_video.py \
  --video_dir /path/to/videos \
  --checkpoint_path /path/to/final_model.pth \
  --sample_idx 0 \
  --output_dir /path/to/gradcam_outputs \
  --save_format both
```

Select a file instead of index:

```bash
python scripts/make_gradcam_video.py \
  --video_dir /path/to/videos \
  --checkpoint_path /path/to/final_model.pth \
  --filename 109_1_FDS.mp4 \
  --output_dir /path/to/gradcam_outputs
```

Optional Grad-CAM flags:
- `--target_layer layer4.1.conv2` to override default target layer.
- `--class_idx <int>`:
  - binary head: `0` (non-seizure) or `1` (seizure)
  - multiclass head: class index to explain
- `--use_data_parallel` to wrap model in `DataParallel` if multiple GPUs are available.
- `--save_format {gif,mp4,both}`.
- `--cam_threshold <float>` to suppress low CAM activations in overlay (e.g., `0.5`).
- `--cam_gamma <float>` to sharpen focus by reweighting CAM (e.g., `2.0`).

Binary class-specific examples:

```bash
# Seizure-focused CAM (class 1)
python scripts/make_gradcam_video.py \
  --video_dir /path/to/videos \
  --checkpoint_path /path/to/final_model.pth \
  --filename 109_1_FDS.mp4 \
  --class_idx 1 \
  --output_dir /path/to/gradcam_outputs/class1

# Non-seizure-focused CAM (class 0)
python scripts/make_gradcam_video.py \
  --video_dir /path/to/videos \
  --checkpoint_path /path/to/final_model.pth \
  --filename 109_1_FDS.mp4 \
  --class_idx 0 \
  --output_dir /path/to/gradcam_outputs/class0
```

Multi-layer comparison (bash loop):

```bash
layers=(
  stem.0
  layer1.0.conv2 layer1.1.conv2
  layer2.0.conv2 layer2.1.conv2
  layer3.0.conv2 layer3.1.conv2
  layer4.0.conv2 layer4.1.conv2
)

for layer in "${layers[@]}"; do
  safe_layer="${layer//./_}"
  python scripts/make_gradcam_video.py \
    --video_dir /path/to/videos \
    --checkpoint_path /path/to/final_model.pth \
    --filename 109_1_FDS.mp4 \
    --target_layer "$layer" \
    --class_idx 1 \
    --output_dir "/path/to/gradcam_outputs/${safe_layer}"
done
```

## Output Files

Typical outputs in `--output_dir`:

- `best_fold1.pth`, ..., `best_foldK.pth`
- `fold{fold}_epoch_metrics.csv/.xlsx` (per-epoch train/val metrics at threshold 0.5)
- `fold{fold}_validation_metrics.csv/.xlsx`
- `cv_fold_metrics.csv/.xlsx`
- `cv_summary.csv`
- `youden_thresholds.npy`
- `final_threshold.json`
- `final_model.pth`
- `final_train_history.csv/.xlsx`
- `internal_dataset_manifest.csv/.xlsx`
- `external_dataset_manifest.csv/.xlsx`
- `external_video_predictions.csv/.xlsx` (per-video probability, predicted label, correctness)
- `external_misclassified_videos.csv/.xlsx` (subset of failed predictions at primary threshold)
- `external_metrics_by_threshold.csv/.xlsx`
- `external_eval_summary.csv`
- Grad-CAM outputs: `<video_stem>_cam.pt`
- Grad-CAM overlays: `<video_stem>_gradcam.gif` and/or `<video_stem>_gradcam.mp4`

Saved metric tables now include confusion-matrix counts:
- `tn`, `fp`, `fn`, `tp`

Plots in `--plot_dir`:
- `loss_curve_fold{fold}.png`

## Notes

- The dataset loader trims each video by removing the first `118` seconds and last `120` seconds.
- To reduce memory pressure, it decodes only this trimmed temporal window instead of the full video.
- If a clip is too short after trimming, loading raises an error.
- Training pipeline is binary classification with a single-logit head.
