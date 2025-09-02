from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif

def select_features_with_variance_threshold(X, threshold=0.01):
    vt = VarianceThreshold(threshold=threshold)
    X_var = vt.fit_transform(X)
    return X_var, X.columns[vt.get_support()]

def select_k_best_features(X, y, k=50):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, X.columns[selector.get_support()]

def compute_mutual_information(X, y):
    mi_scores = mutual_info_classif(X, y, random_state=42)
    return mi_scores, X.columns