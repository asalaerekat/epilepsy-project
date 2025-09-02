from typing import Any, Dict
import pandas as pd

def compute_temporal_derivatives(df: pd.DataFrame, fps: int = 25, extra_spatial_feat: list = None) -> pd.DataFrame:
    if extra_spatial_feat is None:
        extra_spatial_feat = []

    df = df.copy()
    
    landmark_cols = [col for col in df.columns if (col.endswith('_x') or col.endswith('_y')) and 'likelihood' not in col]
    
    feature_cols = landmark_cols + extra_spatial_feat
    
    for col in feature_cols:
        df[f'velocity_{col}'] = df.groupby('video_id')[col].diff() * fps
        df[f'acceleration_{col}'] = df.groupby('video_id')[f'velocity_{col}'].diff() * fps
        df[f'jerk_{col}'] = df.groupby('video_id')[f'acceleration_{col}'].diff() * fps
        
    return df

def compute_motion_magnitude(df: pd.DataFrame) -> pd.DataFrame:
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