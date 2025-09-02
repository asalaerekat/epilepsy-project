from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(y_true, y_pred):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1-score": f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics

def print_evaluation_metrics(metrics):
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

def confusion_matrix_analysis(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()
    
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"True Negatives: {TN}, False Positives: {FP}, False Negatives: {FN}, True Positives: {TP}")