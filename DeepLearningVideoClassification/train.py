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
from sklearn.model_selection import KFold
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
    train_tf = T.Compose([
        T.Resize((128, 171)),
        T.RandomResizedCrop((112, 112), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.RandomGrayscale(0.1),
        T.GaussianBlur((3, 3), sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
    ])
    val_tf = T.Compose([
        T.Resize((128, 171)),
        T.CenterCrop((112, 112)),
        T.ToTensor(),
        T.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
    ])
    return VideoAugmentation(train_tf), VideoAugmentation(val_tf)


def build_dataset(video_dir, num_frames, transform, file_list=None, dataset_name="dataset"):
    ds = VideoDataset(video_dir, num_frames=num_frames, transform=transform, file_list=file_list)
    if len(ds) == 0:
        raise ValueError(
            f"No videos found in {dataset_name} directory: {video_dir}. "
            "Expected filename suffixes matching class labels."
        )
    return ds


def compute_pos_weight(base_dataset, files, device):
    labels = [base_dataset._get_label(fn) for fn in files]
    counts = Counter(labels)
    neg = counts.get(0, 0)
    pos = counts.get(1, 0)

    if neg == 0 or pos == 0:
        print("Warning: one class is missing in this split. Falling back to pos_weight=1.0")
        return torch.tensor([1.0], dtype=torch.float32, device=device)

    return torch.tensor([neg / pos], dtype=torch.float32, device=device)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

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

    return total_loss / len(loader.dataset)


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


def build_optimizer_and_scheduler(model, args):
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    return optimizer, scheduler


def run_kfold(args, device, seed, train_transform, val_transform):
    internal_base_ds = build_dataset(
        args.video_dir, args.num_frames, transform=None, dataset_name="internal"
    )
    all_files = internal_base_ds.video_files

    if len(all_files) < args.k_folds:
        raise ValueError(
            f"k_folds={args.k_folds} is larger than the number of internal videos={len(all_files)}."
        )

    pd.DataFrame({"filename": all_files}).to_excel(
        os.path.join(args.output_dir, "internal_dataset_manifest.xlsx"), index=False
    )

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=seed)
    youden_thresholds = []

    for fold, (tr_idx, cv_idx) in enumerate(kf.split(all_files), start=1):
        print(f"\n=== Fold {fold}/{args.k_folds} ===")
        tr_files = [all_files[i] for i in tr_idx]
        cv_files = [all_files[i] for i in cv_idx]

        tr_ds = build_dataset(
            args.video_dir, args.num_frames, transform=train_transform, file_list=tr_files
        )
        cv_ds = build_dataset(
            args.video_dir, args.num_frames, transform=val_transform, file_list=cv_files
        )

        tr_loader = DataLoader(
            tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        cv_loader = DataLoader(
            cv_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        pos_weight = compute_pos_weight(internal_base_ds, tr_files, device)
        model = get_model(args.model, args.pretrained, 1).to(device)
        optimizer, scheduler = build_optimizer_and_scheduler(model, args)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        best_val_probs_np = None
        best_val_labels_np = None
        train_losses = []
        val_losses = []

        for ep in range(1, args.epochs + 1):
            avg_train_loss = train_epoch(model, tr_loader, criterion, optimizer, device)
            avg_val_loss, _, val_probs_np, val_labels_np = eval_epoch(
                model, cv_loader, criterion, device
            )

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            m05_val = compute_metrics_at_threshold(val_probs_np, val_labels_np, threshold=0.5)
            val_auc = compute_auc(val_probs_np, val_labels_np)
            val_auprc = compute_auprc(val_probs_np, val_labels_np)

            print(
                f"[F{fold}E{ep}] "
                f"tr_loss={avg_train_loss:.3f} | "
                f"val_loss={avg_val_loss:.3f} | "
                f"val_acc(0.5)={m05_val['accuracy']:.3f} | "
                f"val_prec(0.5)={m05_val['precision']:.3f} | "
                f"val_rec(0.5)={m05_val['recall']:.3f} | "
                f"val_f1(0.5)={m05_val['f1']:.3f} | "
                f"val_AUROC={val_auc:.3f} | "
                f"val_AUPRC={val_auprc:.3f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            scheduler.step()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_probs_np = val_probs_np
                best_val_labels_np = val_labels_np
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_fold{fold}.pth"))

        plot_loss_curves(train_losses, val_losses, fold, args.plot_dir)

        youden_t = compute_youden_threshold(best_val_labels_np, best_val_probs_np)
        youden_thresholds.append(youden_t)
        print(f"[Fold {fold}] Youden optimal threshold = {youden_t:.3f}")

        fold_thresholds = sorted(set([0.10, 0.25, youden_t, 0.50, 0.75, 0.90]))
        val_results = []
        for threshold in fold_thresholds:
            m = compute_metrics_at_threshold(
                best_val_probs_np, best_val_labels_np, threshold=threshold
            )
            m["threshold"] = threshold
            m["fold"] = fold
            m["youden"] = youden_t
            val_results.append(m)
            print(
                f"  [Fold {fold}] Th={threshold:.2f} | "
                f"Acc={m['accuracy']:.3f} | "
                f"Prec={m['precision']:.3f} | "
                f"Rec={m['recall']:.3f} | "
                f"Spec={m['specificity']:.3f} | "
                f"F1={m['f1']:.3f}"
            )

        pd.DataFrame(val_results).to_excel(
            os.path.join(args.output_dir, f"fold{fold}_validation_metrics.xlsx"), index=False
        )

    np.save(os.path.join(args.output_dir, "youden_thresholds.npy"), np.array(youden_thresholds))
    print("\nSaved fold checkpoints and per-fold Youden thresholds.")


def run_final_train(args, device, train_transform):
    internal_ds = build_dataset(
        args.video_dir, args.num_frames, transform=train_transform, dataset_name="internal"
    )

    pd.DataFrame({"filename": internal_ds.video_files}).to_excel(
        os.path.join(args.output_dir, "internal_dataset_manifest.xlsx"), index=False
    )

    train_loader = DataLoader(
        internal_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    pos_weight = compute_pos_weight(internal_ds, internal_ds.video_files, device)
    model = get_model(args.model, args.pretrained, 1).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_train_loss = float("inf")
    history = []
    for ep in range(1, args.epochs + 1):
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history.append({"epoch": ep, "train_loss": avg_train_loss})
        print(
            f"[FinalTrain E{ep}] train_loss={avg_train_loss:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        scheduler.step()

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))

    pd.DataFrame(history).to_excel(
        os.path.join(args.output_dir, "final_train_history.xlsx"), index=False
    )
    print("Saved final model to final_model.pth")


def resolve_eval_checkpoint(args):
    fold_ckpt = os.path.join(args.output_dir, f"best_fold{args.best_fold}.pth")
    final_ckpt = os.path.join(args.output_dir, "final_model.pth")

    if os.path.exists(fold_ckpt):
        return fold_ckpt
    if os.path.exists(final_ckpt):
        return final_ckpt

    raise FileNotFoundError(
        "No evaluation checkpoint found. Expected either "
        f"{fold_ckpt} or {final_ckpt}."
    )


def run_external_eval(args, device, val_transform):
    if not args.external_video_dir:
        raise ValueError("--external_video_dir is required when split_mode=external_eval")

    external_ds = build_dataset(
        args.external_video_dir,
        args.num_frames,
        transform=val_transform,
        dataset_name="external",
    )
    pd.DataFrame({"filename": external_ds.video_files}).to_excel(
        os.path.join(args.output_dir, "external_dataset_manifest.xlsx"), index=False
    )

    external_loader = DataLoader(
        external_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ckpt_path = resolve_eval_checkpoint(args)
    print(f"Evaluating checkpoint: {ckpt_path}")

    model = get_model(args.model, args.pretrained, 1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    criterion = nn.BCEWithLogitsLoss()
    loss, _, probs_np, labels_np = eval_epoch(model, external_loader, criterion, device)
    auroc = compute_auc(probs_np, labels_np)
    auprc = compute_auprc(probs_np, labels_np)
    print(f"External Loss: {loss:.3f} | AUROC: {auroc:.3f} | AUPRC: {auprc:.3f}")

    thresholds = [0.10, 0.25, 0.50, 0.75, 0.90]
    youden_path = os.path.join(args.output_dir, "youden_thresholds.npy")
    if os.path.exists(youden_path):
        youden_thresholds = np.load(youden_path)
        if 1 <= args.best_fold <= len(youden_thresholds):
            thresholds.append(float(youden_thresholds[args.best_fold - 1]))

    thresholds = sorted(set(thresholds))
    results = []
    for threshold in thresholds:
        m = compute_metrics_at_threshold(probs_np, labels_np, threshold=threshold)
        m["threshold"] = threshold
        m["auroc"] = auroc
        m["auprc"] = auprc
        m["checkpoint"] = ckpt_path
        results.append(m)
        print(
            f"External Th={threshold:.2f} | "
            f"Acc={m['accuracy']:.3f} | "
            f"Prec={m['precision']:.3f} | "
            f"Rec={m['recall']:.3f} | "
            f"Spec={m['specificity']:.3f} | "
            f"F1={m['f1']:.3f}"
        )

    pd.DataFrame(results).to_excel(
        os.path.join(args.output_dir, "external_metrics_by_threshold.xlsx"), index=False
    )
    print("Saved external evaluation metrics to external_metrics_by_threshold.xlsx")


def main():
    args = get_args()
    seed = 42

    if args.eval_only:
        print("`--eval_only` is deprecated; overriding split_mode to `external_eval`.")
        args.split_mode = "external_eval"

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    set_seed(seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    train_transform, val_transform = build_transforms()

    if args.split_mode == "kfold":
        run_kfold(args, device, seed, train_transform, val_transform)
    elif args.split_mode == "final_train":
        run_final_train(args, device, train_transform)
    elif args.split_mode == "external_eval":
        run_external_eval(args, device, val_transform)
    else:
        raise ValueError(f"Unsupported split_mode: {args.split_mode}")


if __name__ == "__main__":
    main()
