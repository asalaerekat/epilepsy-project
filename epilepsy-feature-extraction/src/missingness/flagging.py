from typing import Any, Dict
import pandas as pd

def flag_missing_frames(df: pd.DataFrame, likelihood_threshold: float = 0.6, missingness_threshold: float = 0.5) -> pd.DataFrame:
    """
    For each frame in the DataFrame, calculate the missing rate based on the likelihood values.
    A keypoint is considered missing if its likelihood is below likelihood_threshold.
    Frames with missing rate above missingness_threshold are flagged.

    Args:
        df (pd.DataFrame): DataFrame with columns for each keypoint in the format: 
                           "BodyPart_x", "BodyPart_y", "BodyPart_likelihood".
        likelihood_threshold (float): Threshold below which a keypoint is considered low confidence.
        missingness_threshold (float): Fraction of missing keypoints in a frame above which the frame is flagged.
    
    Returns:
        pd.DataFrame: The input DataFrame with three new columns:
            - "missing_count": Number of keypoints flagged as missing in that frame.
            - "missing_rate": Fraction of keypoints missing in that frame.
            - "flag_missing": Boolean flag; True if missing_rate > missingness_threshold.
    """
    likelihood_cols = [col for col in df.columns if col.endswith('_likelihood')]
    body_parts = [col.rsplit('_', 1)[0] for col in likelihood_cols]
    body_parts = list(set(body_parts))

    total_keypoints = len(body_parts)

    def missing_count(row: pd.Series) -> int:
        count = 0
        for bp in body_parts:
            likelihood_val = row.get(f"{bp}_likelihood", 0)
            if likelihood_val < likelihood_threshold:
                count += 1
        return count

    df['missing_count'] = df.apply(missing_count, axis=1)
    df['missing_rate'] = df['missing_count'] / total_keypoints
    df['flag_missing'] = df['missing_rate'] > missingness_threshold

    return df

def analyze_flagged_frames(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the percentage and count of frames flagged as missing.

    Args:
        df (pd.DataFrame): DataFrame containing a 'flag_missing' column.
    
    Returns:
        dict: A dictionary containing total frames, flagged count, and flagged percentage.
    """
    total_frames = len(df)
    flagged_count = df['flag_missing'].sum() 
    flagged_percent = (flagged_count / total_frames) * 100
    
    return {
        "total_frames": total_frames,
        "flagged_count": flagged_count,
        "flagged_percent": flagged_percent
    }