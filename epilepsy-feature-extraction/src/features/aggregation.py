from typing import Any, Dict
import pandas as pd

def aggregate_features_by_interval(df: pd.DataFrame, fps: int = 25, interval_seconds: float = 1.0) -> pd.DataFrame:
    interval_frames = int(fps * interval_seconds)
    
    df = df.copy()
    df['relative_frame'] = df.groupby('video_id')['frame'].transform(lambda x: x - x.min())
    df['interval'] = df['relative_frame'] // interval_frames

    if 'flag_missing' in df.columns:
        df['flag_missing'] = df['flag_missing'].astype(int)

    metadata_unique_cols = ['event', 'segment']
    metadata_concat_cols = ['frame', 'relative_frame']

    feature_cols = [col for col in df.columns if col not in metadata_unique_cols + metadata_concat_cols + ['interval', 'video_id']]

    def safe_mean(x):
        return np.nan if x.isnull().all() else np.nanmean(x)

    def safe_std(x):
        return np.nan if x.isnull().all() else np.nanstd(x)

    def safe_median(x):
        return np.nan if x.isnull().all() else np.nanmedian(x)

    def safe_quantile(x, q):
        return np.nan if x.isnull().all() else np.nanquantile(x, q)

    def safe_min(x):
        return np.nan if x.isnull().all() else np.nanmin(x)

    def safe_max(x):
        return np.nan if x.isnull().all() else np.nanmax(x)

    agg_dict = {}
    
    for col in feature_cols:
        agg_dict[f"{col}_mean"] = (col, safe_mean)
        agg_dict[f"{col}_std"] = (col, safe_std)
        agg_dict[f"{col}_median"] = (col, safe_median)
        agg_dict[f"{col}_Q1"] = (col, lambda x: safe_quantile(x, 0.25))
        agg_dict[f"{col}_Q3"] = (col, lambda x: safe_quantile(x, 0.75))
        agg_dict[f"{col}_min"] = (col, safe_min)
        agg_dict[f"{col}_max"] = (col, safe_max)

    if 'flag_missing' in df.columns:
        agg_dict['flagged_frame_fraction'] = ('flag_missing', safe_mean)

    for col in metadata_unique_cols:
        agg_dict[col] = (col, lambda x: ','.join(map(str, x.unique())))

    for col in metadata_concat_cols:
        agg_dict[col] = (col, lambda x: ','.join(x.dropna().astype(str)))

    agg_df = df.groupby(['video_id', 'interval']).agg(**agg_dict).reset_index()

    agg_df['left_arm_state'] = agg_df['left_elbow_angle_mean'].apply(classify_arm_state)
    agg_df['right_arm_state'] = agg_df['right_elbow_angle_mean'].apply(classify_arm_state)

    return agg_df

def classify_arm_state(angle: float) -> str:
    if np.isnan(angle):
        return "missing"
    elif angle >= 160:
        return "Straight Arm"
    elif angle >= 120:
        return "Slightly Bent"
    elif angle >= 90:
        return "Moderately Bent"
    elif angle >= 60:
        return "Highly Bent"
    else:
        return "Folded/Closed Position"