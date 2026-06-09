import json
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from config import get_args
from dataset import VideoAugmentation, VideoDataset
from models import get_model
from utils import compute_auc, compute_auprc, compute_metrics_at_threshold, plot_loss_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def build_transforms():
    # New: clip-consistent augmentation. Random crop/flip are sampled once per clip
    # and applied to every frame, preserving temporal coherence for the 3D CNN.
    train_tf = VideoAugmentation(is_train=True)
    val_tf = VideoAugmentation(is_train=False)
    return train_tf, val_tf


def build_dataset(
    video_dir,
    num_frames,
    transform,
    file_list=None,
    dataset_name="dataset",
    sample_mode="center_clip",
    clip_fps=4.0,
):
    ds = VideoDataset(
        video_dir,
        num_frames=num_frames,
        transform=transform,
        file_list=file_list,
        sample_mode=sample_mode,
        clip_fps=clip_fps,
    )
    if len(ds) == 0:
        raise ValueError(
            f"No videos found in {dataset_name} directory: {video_dir}. "
            "Expected filename suffixes matching class labels."
        )
    return ds


def extract_patient_group(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split("_")[0]


def compute_pos_weight(base_dataset, files, device):
    labels = [base_dataset._get_label(fn) for fn in files]
    counts = Counter(labels)
    neg = counts.get(0, 0)
    pos = counts.get(1, 0)

    if neg == 0 or pos == 0:
        print("Warning: one class is missing in this split. Falling back to pos_weight=1.0")
        return torch.tensor([1.0], dtype=torch.float32, device=device)

    return torch.tensor([neg / pos], dtype=torch.float32, device=device)


def train_epoch(model, loader, criterion, optimizer, device, return_probs_labels=False):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()
        out = model(x).squeeze(1)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        if return_probs_labels:
            all_logits.append(out.detach().cpu())
            all_labels.append(y.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    if not return_probs_labels:
        return avg_loss

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return avg_loss, logits, probs, labels


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float()

            out = model(x).squeeze(1)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)

            all_logits.append(out.detach().cpu())
            all_labels.append(y.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return avg_loss, logits, probs, labels


def compute_youden_threshold(labels_np, probs_np):
    try:
        fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
    except ValueError:
        return 0.5

    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def build_optimizer_and_scheduler(model, args, total_epochs=None):
    base_model = model.module if hasattr(model, "module") else model

    param_groups = []
    if hasattr(base_model, "fc") and hasattr(base_model, "layer4"):
        fc_params = [p for p in base_model.fc.parameters() if p.requires_grad]
        layer4_params = [p for p in base_model.layer4.parameters() if p.requires_grad]

        seen_param_ids = {id(p) for p in fc_params + layer4_params}
        other_params = [
            p for p in base_model.parameters()
            if p.requires_grad and id(p) not in seen_param_ids
        ]

        if fc_params:
            param_groups.append({"params": fc_params, "lr": args.learning_rate, "name": "fc"})
        if layer4_params:
            param_groups.append(
                {"params": layer4_params, "lr": args.layer4_learning_rate, "name": "layer4"}
            )
        if other_params:
            param_groups.append(
                {"params": other_params, "lr": args.layer4_learning_rate, "name": "other"}
            )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if trainable_params:
            param_groups.append({"params": trainable_params, "lr": args.learning_rate, "name": "all"})

    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer.")

    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
        )
    elif args.scheduler == "cosine":
        if total_epochs is None:
            total_epochs = args.epochs
        t_max = args.lr_t_max if args.lr_t_max > 0 else total_epochs
        t_max = max(1, int(t_max))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=args.lr_eta_min,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    return optimizer, scheduler


def save_df(df, csv_path, xlsx_path):
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)


def aggregate_clip_probs(probs, aggregation="topk_mean", topk=5):
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        raise ValueError("Cannot aggregate an empty probability array.")

    if aggregation == "mean":
        return float(np.mean(probs))
    if aggregation == "max":
        return float(np.max(probs))
    if aggregation == "topk_mean":
        k = int(max(1, min(topk, probs.size)))
        return float(np.mean(np.sort(probs)[-k:]))
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def binary_cross_entropy_from_probs(probs, labels, eps=1e-7):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    probs = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


def format_lrs(optimizer):
    pieces = []
    for i, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group{i}")
        pieces.append(f"{name}:{group['lr']:.2e}")
    return ", ".join(pieces)


def lr_columns(optimizer):
    columns = {}
    for i, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group{i}")
        safe_name = name.replace(" ", "_")
        columns[f"lr_{safe_name}"] = float(group["lr"])
    return columns


def run_kfold_cv(args, device, train_transform, val_transform):
    internal_base_ds = build_dataset(
        args.video_dir,
        args.num_frames,
        transform=None,
        dataset_name="internal",
        sample_mode=args.val_sample_mode,
        clip_fps=args.clip_fps,
    )
    all_files = sorted(internal_base_ds.video_files)
    groups = np.array([extract_patient_group(fn) for fn in all_files])
    unique_groups = np.unique(groups)

    if len(all_files) < args.k_folds:
        raise ValueError(
            f"k_folds={args.k_folds} is larger than number of internal videos={len(all_files)}."
        )
    if len(unique_groups) < args.k_folds:
        raise ValueError(
            f"k_folds={args.k_folds} is larger than unique patients={len(unique_groups)}."
        )

    manifest_df = pd.DataFrame({"filename": all_files, "patient_group": groups})
    save_df(
        manifest_df,
        os.path.join(args.output_dir, "internal_dataset_manifest.csv"),
        os.path.join(args.output_dir, "internal_dataset_manifest.xlsx"),
    )

    splitter = GroupKFold(n_splits=args.k_folds)
    youden_thresholds = []
    stop_epochs = []
    fold_rows = []

    for fold, (tr_idx, cv_idx) in enumerate(splitter.split(all_files, groups=groups), start=1):
        print(f"\n=== Fold {fold}/{args.k_folds} ===")

        tr_files = [all_files[i] for i in tr_idx]
        cv_files = [all_files[i] for i in cv_idx]
        tr_groups = set(groups[tr_idx].tolist())
        cv_groups = set(groups[cv_idx].tolist())
        if tr_groups.intersection(cv_groups):
            raise RuntimeError(f"Group leakage detected in fold {fold}.")

        tr_ds = build_dataset(
            args.video_dir,
            args.num_frames,
            transform=train_transform,
            file_list=tr_files,
            sample_mode=args.train_sample_mode,
            clip_fps=args.clip_fps,
        )
        cv_ds = build_dataset(
            args.video_dir,
            args.num_frames,
            transform=val_transform,
            file_list=cv_files,
            sample_mode=args.val_sample_mode,
            clip_fps=args.clip_fps,
        )

        tr_loader = DataLoader(
            tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        cv_loader = DataLoader(
            cv_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        pos_weight = compute_pos_weight(internal_base_ds, tr_files, device)
        model = get_model(args.model, args.pretrained, 1).to(device)
        optimizer, scheduler = build_optimizer_and_scheduler(model, args, total_epochs=args.epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[Fold {fold}] LR groups -> {format_lrs(optimizer)}")

        best_val_loss = float("inf")
        best_val_probs = None
        best_val_labels = None
        best_epoch = None
        train_losses = []
        val_losses = []
        epoch_rows = []

        for ep in range(1, args.epochs + 1):
            avg_train_loss, _, train_probs, train_labels = train_epoch(
                model,
                tr_loader,
                criterion,
                optimizer,
                device,
                return_probs_labels=True,
            )
            avg_val_loss, _, val_probs, val_labels = eval_epoch(model, cv_loader, criterion, device)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            m05_train = compute_metrics_at_threshold(train_probs, train_labels, threshold=0.5)
            m05 = compute_metrics_at_threshold(val_probs, val_labels, threshold=0.5)
            val_auc = compute_auc(val_probs, val_labels)
            val_auprc = compute_auprc(val_probs, val_labels)

            print(
                f"[F{fold}E{ep}] tr_loss={avg_train_loss:.3f} | val_loss={avg_val_loss:.3f} | "
                f"tr_TP={m05_train['tp']} tr_TN={m05_train['tn']} tr_FP={m05_train['fp']} tr_FN={m05_train['fn']} | "
                f"val_acc(0.5)={m05['accuracy']:.3f} | val_prec(0.5)={m05['precision']:.3f} | "
                f"val_rec(0.5)={m05['recall']:.3f} | val_f1(0.5)={m05['f1']:.3f} | "
                f"val_TP={m05['tp']} val_TN={m05['tn']} val_FP={m05['fp']} val_FN={m05['fn']} | "
                f"val_AUROC={val_auc:.3f} | val_AUPRC={val_auprc:.3f} | "
                f"lr={format_lrs(optimizer)}"
            )

            epoch_row = {
                "fold": fold,
                "epoch": ep,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_auroc": val_auc,
                "val_auprc": val_auprc,
                "threshold": 0.5,
                "train_accuracy": m05_train["accuracy"],
                "train_precision": m05_train["precision"],
                "train_recall": m05_train["recall"],
                "train_specificity": m05_train["specificity"],
                "train_f1": m05_train["f1"],
                "train_tn": m05_train["tn"],
                "train_fp": m05_train["fp"],
                "train_fn": m05_train["fn"],
                "train_tp": m05_train["tp"],
                "val_accuracy": m05["accuracy"],
                "val_precision": m05["precision"],
                "val_recall": m05["recall"],
                "val_specificity": m05["specificity"],
                "val_f1": m05["f1"],
                "val_tn": m05["tn"],
                "val_fp": m05["fp"],
                "val_fn": m05["fn"],
                "val_tp": m05["tp"],
            }
            epoch_row.update(lr_columns(optimizer))
            epoch_rows.append(epoch_row)

            scheduler.step()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_probs = val_probs
                best_val_labels = val_labels
                best_epoch = ep
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_fold{fold}.pth"))

        if best_epoch is None:
            raise RuntimeError(f"No best epoch found for fold {fold}.")

        fold_epoch_df = pd.DataFrame(epoch_rows)
        save_df(
            fold_epoch_df,
            os.path.join(args.output_dir, f"fold{fold}_epoch_metrics.csv"),
            os.path.join(args.output_dir, f"fold{fold}_epoch_metrics.xlsx"),
        )

        plot_loss_curves(train_losses, val_losses, fold, args.plot_dir)

        fold_auroc = compute_auc(best_val_probs, best_val_labels)
        fold_auprc = compute_auprc(best_val_probs, best_val_labels)
        youden_t = compute_youden_threshold(best_val_labels, best_val_probs)

        stop_epochs.append(int(best_epoch))
        youden_thresholds.append(youden_t)

        print(
            f"[Fold {fold}] stop_epoch={best_epoch} | val_loss={best_val_loss:.3f} | "
            f"AUROC={fold_auroc:.3f} | AUPRC={fold_auprc:.3f} | Youden={youden_t:.3f}"
        )

        fold_thresholds = sorted(set([0.10, 0.25, youden_t, 0.50, 0.75, 0.90]))
        val_rows = []
        for t in fold_thresholds:
            m = compute_metrics_at_threshold(best_val_probs, best_val_labels, threshold=t)
            val_rows.append(
                {
                    "fold": fold,
                    "threshold": t,
                    "youden_threshold": youden_t,
                    "auroc": fold_auroc,
                    "auprc": fold_auprc,
                    "stop_epoch": best_epoch,
                    "val_loss": best_val_loss,
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "specificity": m["specificity"],
                    "f1": m["f1"],
                    "tn": m["tn"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "tp": m["tp"],
                }
            )

        fold_val_df = pd.DataFrame(val_rows)
        save_df(
            fold_val_df,
            os.path.join(args.output_dir, f"fold{fold}_validation_metrics.csv"),
            os.path.join(args.output_dir, f"fold{fold}_validation_metrics.xlsx"),
        )

        fold_rows.append(
            {
                "fold": fold,
                "n_train_videos": len(tr_files),
                "n_val_videos": len(cv_files),
                "n_train_patients": len(tr_groups),
                "n_val_patients": len(cv_groups),
                "stop_epoch": best_epoch,
                "val_loss": best_val_loss,
                "auroc": fold_auroc,
                "auprc": fold_auprc,
                "youden_threshold": youden_t,
            }
        )

    np.save(os.path.join(args.output_dir, "youden_thresholds.npy"), np.array(youden_thresholds))

    fold_metrics_df = pd.DataFrame(fold_rows)
    save_df(
        fold_metrics_df,
        os.path.join(args.output_dir, "cv_fold_metrics.csv"),
        os.path.join(args.output_dir, "cv_fold_metrics.xlsx"),
    )

    summary_rows = []
    for metric in ["stop_epoch", "val_loss", "auroc", "auprc", "youden_threshold"]:
        mean = float(fold_metrics_df[metric].mean())
        std = float(fold_metrics_df[metric].std(ddof=1)) if len(fold_metrics_df) > 1 else 0.0
        summary_rows.append(
            {"metric": metric, "mean": mean, "std": std, "mean_std": f"{mean:.4f}+/-{std:.4f}"}
        )
    cv_summary_df = pd.DataFrame(summary_rows)
    cv_summary_df.to_csv(os.path.join(args.output_dir, "cv_summary.csv"), index=False)

    median_stop_epoch = int(np.rint(np.median(stop_epochs))) if stop_epochs else args.epochs
    median_stop_epoch = max(1, min(args.epochs, median_stop_epoch))
    avg_youden = float(np.mean(youden_thresholds)) if youden_thresholds else 0.5

    print("\n=== CV Summary (mean+/-std) ===")
    for row in summary_rows:
        print(f"{row['metric']}: {row['mean_std']}")
    print(f"Median stop epoch from folds: {median_stop_epoch}")
    print(f"Average Youden threshold from folds: {avg_youden:.4f}")

    return {
        "youden_thresholds": youden_thresholds,
        "stop_epochs": stop_epochs,
        "median_stop_epoch": median_stop_epoch,
        "avg_youden_threshold": avg_youden,
    }


def save_final_threshold(args, cv_results):
    avg_youden = cv_results["avg_youden_threshold"]
    if args.use_avg_youden_threshold:
        threshold = avg_youden
        source = "avg_kfold_youden"
    else:
        threshold = 0.5
        source = "default_0.5"

    payload = {
        "threshold": float(threshold),
        "source": source,
        "avg_kfold_youden_threshold": float(avg_youden),
        "k_folds": int(args.k_folds),
        "stop_epochs": [int(x) for x in cv_results["stop_epochs"]],
        "median_stop_epoch": int(cv_results["median_stop_epoch"]),
    }

    out_path = os.path.join(args.output_dir, "final_threshold.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved final threshold config to {out_path} ({source}={threshold:.4f})")
    return float(threshold), source


def train_final_model(args, device, train_transform, stop_epoch):
    internal_ds = build_dataset(
        args.video_dir,
        args.num_frames,
        transform=train_transform,
        dataset_name="internal",
        sample_mode=args.train_sample_mode,
        clip_fps=args.clip_fps,
    )
    train_loader = DataLoader(
        internal_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    model = get_model(args.model, args.pretrained, 1).to(device)
    pos_weight = compute_pos_weight(internal_ds, internal_ds.video_files, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer, scheduler = build_optimizer_and_scheduler(model, args, total_epochs=stop_epoch)
    print(f"[FinalTrain] LR groups -> {format_lrs(optimizer)}")

    history = []
    for ep in range(1, stop_epoch + 1):
        avg_train_loss, _, train_probs, train_labels = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            return_probs_labels=True,
        )
        m05_train = compute_metrics_at_threshold(train_probs, train_labels, threshold=0.5)
        row = {
            "epoch": ep,
            "train_loss": avg_train_loss,
            "threshold": 0.5,
            "train_accuracy": m05_train["accuracy"],
            "train_precision": m05_train["precision"],
            "train_recall": m05_train["recall"],
            "train_specificity": m05_train["specificity"],
            "train_f1": m05_train["f1"],
            "train_tn": m05_train["tn"],
            "train_fp": m05_train["fp"],
            "train_fn": m05_train["fn"],
            "train_tp": m05_train["tp"],
        }
        row.update(lr_columns(optimizer))
        history.append(row)
        print(
            f"[FinalTrain E{ep}/{stop_epoch}] train_loss={avg_train_loss:.3f} | "
            f"train_TP={m05_train['tp']} train_TN={m05_train['tn']} train_FP={m05_train['fp']} train_FN={m05_train['fn']} | "
            f"lr={format_lrs(optimizer)}"
        )
        scheduler.step()

    model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), model_path)

    history_df = pd.DataFrame(history)
    save_df(
        history_df,
        os.path.join(args.output_dir, "final_train_history.csv"),
        os.path.join(args.output_dir, "final_train_history.xlsx"),
    )

    print(f"Saved final model to {model_path}")
    return model_path


def predict_multiclip_video(model, dataset, filename, args, device):
    clips, label, clip_rows = dataset.get_multiclip(filename, num_clips=args.num_test_clips)
    model.eval()

    all_logits = []
    all_probs = []
    batch_size = max(1, int(args.eval_batch_size))

    with torch.no_grad():
        for start in range(0, clips.shape[0], batch_size):
            x = clips[start:start + batch_size].to(device)
            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits)
            all_logits.extend(logits.detach().cpu().numpy().astype(float).tolist())
            all_probs.extend(probs.detach().cpu().numpy().astype(float).tolist())

    video_prob = aggregate_clip_probs(
        all_probs,
        aggregation=args.aggregation,
        topk=args.topk,
    )

    for row, logit, prob in zip(clip_rows, all_logits, all_probs):
        row["clip_logit"] = float(logit)
        row["clip_prob_class1"] = float(prob)
        row["aggregation"] = args.aggregation
        row["topk"] = int(args.topk)
        row["num_test_clips_requested"] = int(args.num_test_clips)

    return {
        "filename": filename,
        "true_label": int(label),
        "prob_class1": float(video_prob),
        "n_clips": int(len(all_probs)),
        "clip_prob_min": float(np.min(all_probs)),
        "clip_prob_max": float(np.max(all_probs)),
        "clip_prob_mean": float(np.mean(all_probs)),
        "clip_prob_topk_mean": float(aggregate_clip_probs(all_probs, "topk_mean", args.topk)),
    }, clip_rows


def evaluate_external(
    args,
    device,
    val_transform,
    checkpoint_path,
    primary_threshold,
    threshold_source,
):
    if not args.external_video_dir:
        raise ValueError("--external_video_dir is required for external evaluation.")

    # New external evaluation: sample multiple local clips per video, predict each clip,
    # then aggregate clip probabilities into one video-level probability.
    external_ds = build_dataset(
        args.external_video_dir,
        args.num_frames,
        transform=val_transform,
        dataset_name="external",
        sample_mode="center_clip",
        clip_fps=args.clip_fps,
    )

    manifest_df = pd.DataFrame({"filename": external_ds.video_files})
    save_df(
        manifest_df,
        os.path.join(args.output_dir, "external_dataset_manifest.csv"),
        os.path.join(args.output_dir, "external_dataset_manifest.xlsx"),
    )

    model = get_model(args.model, args.pretrained, 1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    video_rows = []
    all_clip_rows = []
    for filename in external_ds.video_files:
        try:
            video_row, clip_rows = predict_multiclip_video(model, external_ds, filename, args, device)
        except Exception as exc:
            raise RuntimeError(f"External multi-clip evaluation failed for {filename}") from exc
        video_rows.append(video_row)
        all_clip_rows.extend(clip_rows)

    per_video_df = pd.DataFrame(video_rows)
    labels = per_video_df["true_label"].to_numpy(dtype=int)
    probs = per_video_df["prob_class1"].to_numpy(dtype=float)
    loss = binary_cross_entropy_from_probs(probs, labels)

    primary_preds = (probs >= float(primary_threshold)).astype(int)
    default_preds = (probs >= 0.5).astype(int)

    per_video_df["true_label_name"] = np.where(
        labels == 0, "PNEE/FDS", "GTC/FocalBilateralTC/ES"
    )
    per_video_df["primary_threshold"] = float(primary_threshold)
    per_video_df["threshold_source"] = threshold_source
    per_video_df["pred_label_primary_threshold"] = primary_preds
    per_video_df["pred_label_name_primary_threshold"] = np.where(
        primary_preds == 0, "PNEE/FDS", "GTC/FocalBilateralTC/ES"
    )
    per_video_df["is_correct_primary_threshold"] = (primary_preds == labels).astype(int)
    per_video_df["pred_label_0_5"] = default_preds
    per_video_df["pred_label_name_0_5"] = np.where(
        default_preds == 0, "PNEE/FDS", "GTC/FocalBilateralTC/ES"
    )
    per_video_df["is_correct_0_5"] = (default_preds == labels).astype(int)
    per_video_df["aggregation"] = args.aggregation
    per_video_df["topk"] = int(args.topk)
    per_video_df["num_test_clips_requested"] = int(args.num_test_clips)
    per_video_df["num_frames"] = int(args.num_frames)
    per_video_df["clip_fps"] = float(args.clip_fps)
    per_video_df["local_clip_duration_sec"] = float(args.num_frames) / float(args.clip_fps)

    save_df(
        per_video_df,
        os.path.join(args.output_dir, "external_video_predictions.csv"),
        os.path.join(args.output_dir, "external_video_predictions.xlsx"),
    )

    clip_df = pd.DataFrame(all_clip_rows)
    save_df(
        clip_df,
        os.path.join(args.output_dir, "external_clip_predictions.csv"),
        os.path.join(args.output_dir, "external_clip_predictions.xlsx"),
    )

    misclassified_df = per_video_df[per_video_df["is_correct_primary_threshold"] == 0].copy()
    save_df(
        misclassified_df,
        os.path.join(args.output_dir, "external_misclassified_videos.csv"),
        os.path.join(args.output_dir, "external_misclassified_videos.xlsx"),
    )

    auroc = compute_auc(probs, labels)
    auprc = compute_auprc(probs, labels)

    thresholds = sorted(set([0.5, float(primary_threshold)]))
    threshold_rows = []
    for t in thresholds:
        m = compute_metrics_at_threshold(probs, labels, threshold=t)
        threshold_rows.append(
            {
                "threshold": t,
                "is_primary_threshold": int(np.isclose(t, primary_threshold)),
                "threshold_source": threshold_source,
                "loss": loss,
                "auroc": auroc,
                "auprc": auprc,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "specificity": m["specificity"],
                "f1": m["f1"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
                "aggregation": args.aggregation,
                "topk": int(args.topk),
                "num_test_clips": int(args.num_test_clips),
                "num_frames": int(args.num_frames),
                "clip_fps": float(args.clip_fps),
            }
        )

    threshold_df = pd.DataFrame(threshold_rows)
    save_df(
        threshold_df,
        os.path.join(args.output_dir, "external_metrics_by_threshold.csv"),
        os.path.join(args.output_dir, "external_metrics_by_threshold.xlsx"),
    )

    primary_row = next(
        row for row in threshold_rows if int(np.isclose(row["threshold"], primary_threshold)) == 1
    )
    summary_df = pd.DataFrame(
        [
            {
                "checkpoint_path": checkpoint_path,
                "threshold": primary_threshold,
                "threshold_source": threshold_source,
                "loss": loss,
                "auroc": auroc,
                "auprc": auprc,
                "accuracy": primary_row["accuracy"],
                "precision": primary_row["precision"],
                "recall": primary_row["recall"],
                "specificity": primary_row["specificity"],
                "f1": primary_row["f1"],
                "tn": primary_row["tn"],
                "fp": primary_row["fp"],
                "fn": primary_row["fn"],
                "tp": primary_row["tp"],
                "aggregation": args.aggregation,
                "topk": int(args.topk),
                "num_test_clips": int(args.num_test_clips),
                "num_frames": int(args.num_frames),
                "clip_fps": float(args.clip_fps),
            }
        ]
    )
    summary_df.to_csv(os.path.join(args.output_dir, "external_eval_summary.csv"), index=False)

    print(
        f"External multi-clip eval | checkpoint={checkpoint_path} | "
        f"threshold={primary_threshold:.4f} ({threshold_source}) | "
        f"clips/video={args.num_test_clips} | aggregation={args.aggregation} | "
        f"loss={loss:.3f} | AUROC={auroc:.3f} | AUPRC={auprc:.3f}"
    )
    return summary_df.iloc[0].to_dict()

def resolve_test_checkpoint(args):
    if args.checkpoint_path is not None:
        ckpt = args.checkpoint_path
    else:
        ckpt = os.path.join(args.output_dir, "final_model.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def resolve_test_threshold(args):
    if args.threshold is not None:
        return float(args.threshold), "cli_override"

    final_threshold_path = os.path.join(args.output_dir, "final_threshold.json")
    if os.path.exists(final_threshold_path):
        with open(final_threshold_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "threshold" in payload:
            return float(payload["threshold"]), "final_threshold.json"

    return 0.5, "default_0.5"


def main():
    args = get_args()
    seed = 42

    if args.eval_batch_size < 1:
        raise ValueError("--eval_batch_size must be >= 1.")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    set_seed(seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    train_transform, val_transform = build_transforms()

    if args.run_mode == "train":
        if not args.external_video_dir:
            raise ValueError(
                "--external_video_dir is required in train mode "
                "(final model is evaluated on external testing set)."
            )

        cv_results = run_kfold_cv(args, device, train_transform, val_transform)
        threshold, threshold_source = save_final_threshold(args, cv_results)
        final_model_path = train_final_model(
            args, device, train_transform, stop_epoch=cv_results["median_stop_epoch"]
        )
        evaluate_external(
            args,
            device,
            val_transform,
            checkpoint_path=final_model_path,
            primary_threshold=threshold,
            threshold_source=threshold_source,
        )

    elif args.run_mode == "test_only":
        if not args.external_video_dir:
            raise ValueError("--external_video_dir is required in test_only mode.")

        checkpoint_path = resolve_test_checkpoint(args)
        threshold, threshold_source = resolve_test_threshold(args)
        evaluate_external(
            args,
            device,
            val_transform,
            checkpoint_path=checkpoint_path,
            primary_threshold=threshold,
            threshold_source=threshold_source,
        )
    else:
        raise ValueError(f"Unsupported run_mode: {args.run_mode}")


if __name__ == "__main__":
    main()
