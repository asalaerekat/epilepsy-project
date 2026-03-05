import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score


def compute_metrics_at_threshold(probs, labels_np, threshold):
    """
    Given:
      - probs: 1D numpy array of model probabilities for the positive class
      - labels_np: 1D numpy array of true {0,1} labels
      - threshold: float between 0 and 1
    Returns a dict with:
      - accuracy, precision, recall, specificity, f1
      - tn, fp, fn, tp
    """
    # Binarize predictions
    preds = (probs >= threshold).astype(int)

    # Basic metrics
    precision = precision_score(labels_np, preds, zero_division=0)
    recall    = recall_score(labels_np, preds, zero_division=0)
    f1        = f1_score(labels_np, preds, zero_division=0)

    # Accuracy
    accuracy = (preds == labels_np).sum() / len(labels_np)

    # Specificity
    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy":    accuracy,
        "precision":   precision,
        "recall":      recall,
        "specificity": specificity,
        "f1":          f1,
        "tn":          int(tn),
        "fp":          int(fp),
        "fn":          int(fn),
        "tp":          int(tp),
    }


def compute_auprc(probs, labels_np):
    try:
        return average_precision_score(labels_np, probs)
    except ValueError:
        return 0.0

def compute_auc(probs, labels_np):
    """
    Returns the AUROC (area under ROC curve) for a set of probabilities and true labels.
    If only one class is present, returns 0.0.
    """
    try:
        return roc_auc_score(labels_np, probs)
    except ValueError:
        return 0.0
        

def plot_loss_curves(train_losses, val_losses, fold, plot_dir):
    """
    Plots training and validation loss curves and saves the plot.
    """
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Loss Curves - Fold {fold}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_dir}/loss_curve_fold{fold}.png")
    plt.close()
    
