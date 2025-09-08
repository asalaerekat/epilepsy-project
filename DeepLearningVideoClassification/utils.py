import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score


#def accuracy(outputs, labels):
#    """
#    Computes the accuracy for a batch.
#    """
#    _, preds = torch.max(outputs, dim=1)
#    corrects = torch.sum(preds == labels.data)
#    return corrects.double() / labels.size(0)


def accuracy(outputs, labels):
    """
    Computes the accuracy for a batch of raw logits in a binary setup.
    outputs: Tensor of shape (B,) — raw logits
    labels:  Tensor of shape (B,) — {0,1} ints or floats
    """
    probs = torch.sigmoid(outputs)
    preds = (probs >= 0.5).long()
    corrects = (preds == labels.long()).sum()
    return corrects.double() / labels.size(0)

#def compute_metrics(outputs, labels):
#    """
#    Computes additional metrics: precision, recall, specificity, F1 score, and AUC.
#    Expects outputs as raw logits.
#    """
#    # Convert outputs to probabilities and get predictions.
#    probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
#    preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
#    labels_np = labels.cpu().numpy()

#    precision = precision_score(labels_np, preds, zero_division=0)
#    recall = recall_score(labels_np, preds, zero_division=0)
#    f1 = f1_score(labels_np, preds, zero_division=0)
#    # AUC requires both classes to be present; if not, default to 0.0
#    try:
#        auc = roc_auc_score(labels_np, probs)
#    except ValueError:
#        auc = 0.0

#    # Calculate specificity from confusion matrix:
#    cm = confusion_matrix(labels_np, preds)
#    if cm.shape == (2, 2):
#        tn, fp, fn, tp = cm.ravel()
#        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#    else:
#        specificity = 0.0

#    return {
#        "precision": precision,
#        "recall": recall,
#        "specificity": specificity,
#        "f1": f1,
#        "auc": auc
#    }


#def compute_metrics(outputs, labels):
#    """
#    Computes precision, recall, specificity, F1, and AUC
#    for raw logits in a binary setup.
#    """
#    # to CPU numpy
#    probs    = torch.sigmoid(outputs).detach().cpu().numpy()
#    preds    = (probs >= 0.5).astype(int)
#    labels_np = labels.detach().cpu().numpy().astype(int)

#    precision = precision_score(labels_np, preds, zero_division=0)
#    recall    = recall_score(labels_np, preds, zero_division=0)
#    f1        = f1_score(labels_np, preds, zero_division=0)
#    try:
#        auc = roc_auc_score(labels_np, probs)
#    except ValueError:
#        auc = 0.0

#    cm = confusion_matrix(labels_np, preds)
#    if cm.shape == (2, 2):
#        tn, fp, fn, tp = cm.ravel()
#        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#    else:
#        specificity = 0.0

#    return {
#        "precision": precision,
#        "recall":    recall,
#        "specificity": specificity,
#        "f1":        f1,
#        "auc":       auc
#    }



def compute_metrics_at_threshold(probs, labels_np, threshold):
    """
    Given:
      - probs: 1D numpy array of model probabilities for the positive class
      - labels_np: 1D numpy array of true {0,1} labels
      - threshold: float between 0 and 1
    Returns a dict with precision, recall, specificity, accuracy, f1.
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
    cm = confusion_matrix(labels_np, preds)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    return {
        "accuracy":    accuracy,
        "precision":   precision,
        "recall":      recall,
        "specificity": specificity,
        "f1":          f1
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
    