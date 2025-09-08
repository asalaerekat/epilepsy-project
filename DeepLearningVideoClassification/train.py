import os
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
import torchvision.transforms as T

from dataset import VideoDataset, VideoAugmentation
from models import get_model
from config import get_args
from utils import (
    compute_metrics_at_threshold,
    compute_auc,
    compute_auprc,
    plot_loss_curves,
)

# ----------------------------
# Modified eval_epoch to return (avg_loss, logits, probs, labels)
def eval_epoch(model, loader, crit, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float()

            out = model(x).squeeze(1)      # raw logits, shape (B,)
            loss = crit(out, y)
            total_loss += loss.item() * x.size(0)

            all_logits.append(out.detach().cpu())
            all_labels.append(y.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)

    logits = torch.cat(all_logits).numpy()                      # shape (N,)
    labels = torch.cat(all_labels).numpy().astype(int)          # shape (N,)
    probs  = torch.sigmoid(torch.from_numpy(logits)).numpy()    # shape (N,)

    return avg_loss, logits, probs, labels
# ----------------------------

def main():
    args = get_args()
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark    = False

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir,   exist_ok=True)

    # ----------------------------
    # Transforms (unchanged)
    train_tf = T.Compose([
        T.Resize((128, 171)),
        T.RandomResizedCrop((112,112), scale=(0.8,1.0), ratio=(0.9,1.1)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.2,0.2,0.2,0.1),
        T.RandomGrayscale(0.1),
        T.GaussianBlur((3,3), sigma=(0.1,2.0)),
        T.ToTensor(),
        T.Normalize([0.43216,0.394666,0.37645],
                    [0.22803,0.22145,0.216989]),
    ])
    val_tf = T.Compose([
        T.Resize((128, 171)),
        T.CenterCrop((112,112)),
        T.ToTensor(),
        T.Normalize([0.43216,0.394666,0.37645],
                    [0.22803,0.22145,0.216989]),
    ])

    train_transform = VideoAugmentation(train_tf)
    val_transform   = VideoAugmentation(val_tf)
    # ----------------------------

    # Load the full dataset and define train/val/test splits by prefix
    full_ds = VideoDataset(args.video_dir,
                           num_frames=args.num_frames,
                           transform=None)
    files = full_ds.video_files

    test_prefixes = {
        '109_1', '113_1', '117_1', '122_1', '124_1', '126_1', '135_1',
        '18_1',  '20_1',  '27_1',  '32_1',  '3_1',   '43_1',  '48_1',
        '49_1',  '51_1',  '66_1',  '7_4',   '90_1',  '96_1',  '9_4'
    }

    test_ids     = [i for i, f in enumerate(files) if any(f.startswith(p) for p in test_prefixes)]
    trainval_ids = [i for i in range(len(files)) if i not in test_ids]

    trainval_ds = Subset(full_ds, trainval_ids)
    test_ds     = Subset(full_ds, test_ids)

    test_ds.dataset.transform = val_transform

    # Save the test manifest (unchanged)
    pd.DataFrame({"filename": [files[i] for i in test_ids]}) \
      .to_excel(os.path.join(args.output_dir, "test_split.xlsx"), index=False)

    # If eval_only, skip CV and go directly to test set
    if args.eval_only:
        print("\n=== Running evaluation only on test set ===")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=1)

        model = get_model(args.model, args.pretrained, 1).to(device)
        ckpt = os.path.join(args.output_dir, f"best_fold{args.best_fold}.pth")
        model.load_state_dict(torch.load(ckpt, map_location=device))

        criterion = nn.BCEWithLogitsLoss()
        test_loss, logits_np, probs_np, labels_np = eval_epoch(
            model, test_loader, criterion, device
        )

        # Compute AUROC/AUPRC
        auroc = compute_auc(probs_np, labels_np)
        auprc = compute_auprc(probs_np, labels_np)
        print(f"Test Loss: {test_loss:.3f} | AUROC: {auroc:.3f} | AUPRC: {auprc:.3f}")

        # Load saved per-fold Youden thresholds
        youden_path = os.path.join(args.output_dir, "youden_thresholds.npy")
        if os.path.exists(youden_path):
            youden_thresholds = np.load(youden_path)
            # Extract the Youden threshold for this fold (1-indexed in best_fold)
            youden_t = float(youden_thresholds[args.best_fold - 1])
            print(f"Applying Youden’s J threshold from fold {args.best_fold}: {youden_t:.3f}")
        else:
            youden_t = 0.5
            print("Youden thresholds file not found—defaulting to 0.50")

        # Evaluate at default 0.5 and Youden’s J
        for t in [0.50, youden_t]:
            m = compute_metrics_at_threshold(probs_np, labels_np, threshold=t)
            print(f"--- Threshold = {t:.2f} ---")
            print(f"Accuracy   : {m['accuracy']:.3f} | Precision : {m['precision']:.3f} | "
                  f"Recall : {m['recall']:.3f} | Specificity: {m['specificity']:.3f} | "
                  f"F1 : {m['f1']:.3f}")

        return

    # ----------------------------
    # --- K-fold training + validation with Youden’s J in each fold ---
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=seed)

    # To store each fold’s Youden threshold
    youden_thresholds = []

    for fold, (tr_idx, cv_idx) in enumerate(kf.split(trainval_ds), start=1):
        print(f"\n=== Fold {fold}/{args.k_folds} ===")

        # Create training and validation subsets for this fold
        tr_sub = Subset(trainval_ds, tr_idx)
        cv_sub = Subset(trainval_ds, cv_idx)

        tr_sub.dataset.transform = train_transform
        cv_sub.dataset.transform = val_transform

        tr_loader = DataLoader(tr_sub, batch_size=args.batch_size,
                               shuffle=True, num_workers=1)
        cv_loader = DataLoader(cv_sub, batch_size=args.batch_size,
                               shuffle=False, num_workers=1)

        # Compute pos_weight for BCE on the training split of this fold
        abs_tr_idxs = [trainval_ids[i] for i in tr_idx]
        train_labels = [full_ds._get_label(files[i]) for i in abs_tr_idxs]
        counts = Counter(train_labels)
        neg, pos = counts[0], counts[1]
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)

        # Initialize model, optimizer, criterion, scheduler
        model = get_model(args.model, args.pretrained, 1).to(device)
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

        best_loss = float("inf")
        train_losses, val_losses = [], []

        for ep in range(1, args.epochs + 1):
            # --- Train for one epoch ---
            model.train()
            total_loss = 0.0
            all_logits = []
            all_labels = []

            for x, y in tr_loader:
                x = x.to(device)
                y = y.to(device).float()

                optimizer.zero_grad()
                out = model(x).squeeze(1)      # raw logits
                loss = criterion(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                all_logits.append(out.detach().cpu())
                all_labels.append(y.detach().cpu())

            avg_train_loss = total_loss / len(tr_loader.dataset)
            train_losses.append(avg_train_loss)

            # --- Validate on this fold’s validation set ---
            model.eval()
            val_loss = 0.0
            val_logits = []
            val_labels = []

            with torch.no_grad():
                for x_val, y_val in cv_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device).float()

                    out_val = model(x_val).squeeze(1)  # raw logits
                    loss_val = criterion(out_val, y_val)
                    val_loss += loss_val.item() * x_val.size(0)

                    val_logits.append(out_val.detach().cpu())
                    val_labels.append(y_val.detach().cpu())

            avg_val_loss = val_loss / len(cv_loader.dataset)
            val_losses.append(avg_val_loss)

            # Concatenate all validation logits and labels for this epoch
            val_logits_np = torch.cat(val_logits).numpy()
            val_labels_np = torch.cat(val_labels).numpy().astype(int)
            val_probs_np  = torch.sigmoid(torch.from_numpy(val_logits_np)).numpy()

            # Compute metrics on validation at threshold=0.5
            m05_val = compute_metrics_at_threshold(val_probs_np, val_labels_np, threshold=0.5)

            # Compute AUROC & AUPRC on validation
            val_auc   = compute_auc(val_probs_np, val_labels_np)
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

            # Save checkpoint if this epoch’s validation loss is lowest
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, f"best_fold{fold}.pth"))

        # Plot and save loss curves for this fold
        plot_loss_curves(train_losses, val_losses, fold, args.plot_dir)

        # ----------------------------
        # === After finishing CV‐fold training, compute Youden’s J on validation ===
        fpr, tpr, thresh_arr = roc_curve(val_labels_np, val_probs_np)
        j_scores = tpr - fpr
        idx_opt = np.argmax(j_scores)
        youden_t = float(thresh_arr[idx_opt])
        youden_thresholds.append(youden_t)

        print(f"[Fold {fold}] Youden’s J optimal threshold = {youden_t:.3f}")

        # Build a list of thresholds for this fold (including fold’s Youden + some fixed cutoffs)
        fold_thresholds = sorted([0.10, 0.25, youden_t, 0.50, 0.75, 0.90])

        # Compute metrics for each threshold on this fold’s validation set
        val_results = []
        for t in fold_thresholds:
            m = compute_metrics_at_threshold(val_probs_np, val_labels_np, threshold=t)
            m["threshold"] = t
            m["fold"]      = fold
            m["youden"]    = youden_t
            val_results.append(m)

            print(
                f"  → [Fold {fold}] Th={t:.2f} | "
                f"Acc={m['accuracy']:.3f} | "
                f"Prec={m['precision']:.3f} | "
                f"Rec={m['recall']:.3f} | "
                f"Spec={m['specificity']:.3f} | "
                f"F1={m['f1']:.3f}"
            )

        # Save this fold’s validation‐threshold metrics to Excel
        val_results_df = pd.DataFrame(val_results)
        val_results_df.to_excel(
            os.path.join(args.output_dir, f"fold{fold}_validation_metrics.xlsx"),
            index=False
        )
        print(f"[Fold {fold}] validation metrics saved to fold{fold}_validation_metrics.xlsx\n")
        # ----------------------------

    # Save the array of per‐fold Youden’s J thresholds
    youden_array = np.array(youden_thresholds)
    np.save(os.path.join(args.output_dir, "youden_thresholds.npy"), youden_array)

    # ----------------------------
    # === After all folds: select Youden from the best_fold and run final test evaluation ===

    # Extract the Youden threshold for the specified best_fold (1-indexed)
    best_fold = args.best_fold
    final_youden = float(youden_array[best_fold - 1])
    print(f"Fold {best_fold} Youden’s J threshold (to use on test) = {final_youden:.3f}\n")

    # Choose final list of 6 cut‐offs (including the fold’s Youden)
    final_thresholds = sorted([0.10, 0.25, final_youden, 0.50, 0.75, 0.90])

    # --- Final evaluation on hold‐out test set ---
    print("=== Final evaluation on test set ===")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=1)

    final_model = get_model(args.model, args.pretrained, 1).to(device)
    ckpt = os.path.join(args.output_dir, f"best_fold{best_fold}.pth")
    final_model.load_state_dict(torch.load(ckpt, map_location=device))

    final_crit = nn.BCEWithLogitsLoss()
    test_loss, test_logits_np, test_probs_np, test_labels_np = eval_epoch(
        final_model, test_loader, final_crit, device
    )

    # Compute AUROC and AUPRC on test
    test_auroc = compute_auc(test_probs_np, test_labels_np)
    test_auprc = compute_auprc(test_probs_np, test_labels_np)

    print(f"Test Loss: {test_loss:.3f} | Test AUROC: {test_auroc:.3f} | Test AUPRC: {test_auprc:.3f}")

    # Compute metrics at each final threshold on test
    test_results = []
    for t in final_thresholds:
        m = compute_metrics_at_threshold(test_probs_np, test_labels_np, threshold=t)
        m["threshold"] = t
        m["auroc"]     = test_auroc
        m["auprc"]     = test_auprc
        test_results.append(m)

        print(
            f"--- Test Th = {t:.2f} ---\n"
            f"Accuracy   : {m['accuracy']:.3f} | "
            f"Precision  : {m['precision']:.3f} | "
            f"Recall     : {m['recall']:.3f} | "
            f"Specificity: {m['specificity']:.3f} | "
            f"F1‐score   : {m['f1']:.3f}\n"
        )

    # Save test‐set metrics to Excel
    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_excel(
        os.path.join(args.output_dir, "test_metrics_by_threshold.xlsx"),
        index=False
    )
    print("Test‐set threshold metrics saved to test_metrics_by_threshold.xlsx")

if __name__ == "__main__":
    main()