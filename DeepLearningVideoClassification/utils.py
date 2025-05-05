import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def accuracy(outputs, labels):
    """
    Computes the accuracy for a batch.
    """
    _, preds = torch.max(outputs, dim=1)
    corrects = torch.sum(preds == labels.data)
    return corrects.double() / labels.size(0)

def compute_metrics(outputs, labels):
    """
    Computes additional metrics: precision, recall, specificity, F1 score, and AUC.
    Expects outputs as raw logits.
    """
    # Convert outputs to probabilities and get predictions.
    probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
    preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
    labels_np = labels.cpu().numpy()

    precision = precision_score(labels_np, preds, zero_division=0)
    recall = recall_score(labels_np, preds, zero_division=0)
    f1 = f1_score(labels_np, preds, zero_division=0)
    # AUC requires both classes to be present; if not, default to 0.0
    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        auc = 0.0

    # Calculate specificity from confusion matrix:
    cm = confusion_matrix(labels_np, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc
    }

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
    