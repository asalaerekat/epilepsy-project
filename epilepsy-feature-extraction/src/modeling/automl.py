from h2o import H2OFrame, H2OAutoML
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

def train_h2o_automl(X_train, y_train, video_ids_train, target_col, n_splits=5, max_models=20, max_runtime_secs=3600, seed=42, h2o_mem="8G", balance_classes=False):
    h2o.init(max_mem_size=h2o_mem)

    df = X_train.copy()
    df[target_col] = y_train.values
    df["video_id"] = video_ids_train.values
    df["custom_fold"] = -1

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(sgkf.split(df, df[target_col], groups=df["video_id"])):
        df.iloc[val_idx, df.columns.get_loc("custom_fold")] = fold

    for fold in range(n_splits):
        train_vids = set(df.loc[df["custom_fold"] != fold, "video_id"])
        valid_vids = set(df.loc[df["custom_fold"] == fold, "video_id"])
        assert train_vids.isdisjoint(valid_vids), f"Leakage in fold {fold}"

    counts = df[target_col].value_counts()
    ratios = (counts.max() / counts).to_dict()
    df["weight"] = df[target_col].map(ratios)

    hf = H2OFrame(df.drop(columns="video_id"))
    hf["custom_fold"] = hf["custom_fold"].asfactor()
    hf[target_col] = hf[target_col].asfactor()
    hf["weight"] = hf["weight"]

    aml = H2OAutoML(max_models=max_models, max_runtime_secs=max_runtime_secs, seed=seed, balance_classes=balance_classes)

    feature_cols = [c for c in hf.columns if c not in {target_col, "custom_fold", "weight"}]
    aml.train(x=feature_cols, y=target_col, training_frame=hf, fold_column="custom_fold", weights_column="weight")

    return aml

def evaluate_h2o_model(aml, X_test, y_test, target='numeric_event'):
    best_model = aml.leader
    test = H2OFrame(pd.concat([X_test, y_test], axis=1))
    test[target] = test[target].asfactor()
    
    predictions = best_model.predict(test)
    
    pred_df = predictions.as_data_frame()
    y_pred = pred_df['predict'].astype(int)
    y_prob = pred_df['p1']
    
    epsilon = 1e-15  
    logits = np.log(np.clip(y_prob, epsilon, 1 - epsilon) / np.clip(1 - y_prob, epsilon, 1 - y_prob))
    
    y_test_ = test[target].as_data_frame().astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()

    metrics = {
        "Accuracy": accuracy_score(y_test_, y_pred),
        "ROC-AUC": roc_auc_score(y_test_, y_prob),
        "Precision": precision_score(y_test_, y_pred, pos_label=1),
        "Recall": recall_score(y_test_, y_pred, pos_label=1),
        "Specificity": TN / (TN + FP) if (TN + FP) > 0 else 0,
        "F1-score": f1_score(y_test_, y_pred, pos_label=1)
    }
    
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return best_model, test, metrics, y_pred, y_prob, logits