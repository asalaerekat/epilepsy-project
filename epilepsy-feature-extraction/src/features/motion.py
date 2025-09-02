from typing import Any, Dict
import numpy as np
import pandas as pd

def compute_motion_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts (x, y) velocity, acceleration, jerk into single magnitude features
    for both mean and std, and drops raw x/y components.
    
    Args:
        df (pd.DataFrame): DataFrame containing motion features.
    
    Returns:
        pd.DataFrame: DataFrame with magnitude features added and raw components dropped.
    """
    new_cols = {}
    processed_features = []

    keypoints = [
        "left_arm", "right_arm", "left_fingers_avg", "right_fingers_avg", "avg_finger", "body_center", 
        "LeftEye", "RightEye", "Nose", "Chin", "RightWrist", "LeftWrist",
        "RightShoulder", "LeftShoulder", "RightElbow", "LeftElbow",
        "RightHip", "LeftHip", "RightKnee", "LeftKnee", "RightAnkle", "LeftAnkle",
        "RightThumb", "RightIndexFinger", "RightMiddleFinger", "RightRingFinger", "RightPinky", 
        "LeftThumb", "LeftIndexFinger", "LeftMiddleFinger", "LeftRingFinger", "LeftPinky",
        "mid_shoulder"
    ]

    for keypoint in keypoints:
        for motion in ["velocity", "acceleration", "jerk"]:
            for stat in ["mean", "std"]:
                x_col = f"{motion}_{keypoint}_x_{stat}"
                y_col = f"{motion}_{keypoint}_y_{stat}"
                mag_col = f"{motion}_{keypoint}_{stat}_mag"

                if x_col in df.columns and y_col in df.columns:
                    new_cols[mag_col] = np.sqrt(df[x_col]**2 + df[y_col]**2)
                    processed_features.extend([x_col, y_col])

    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    df = df.drop(columns=processed_features)

    return df