from typing import Dict, Any
import numpy as np
import pandas as pd

def compute_head_size(landmarks: Dict[str, Any]) -> float:
    valid_landmarks = {name: coord for name, coord in landmarks.items() if coord is not None}
    coords = np.array(list(valid_landmarks.values()))
    
    if len(coords) < 2:
        return None

    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)

    return np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

def compute_median_head_size_per_video(df: pd.DataFrame, valid_threshold: float = 0.5) -> Dict[str, float]:
    video_head_sizes = {}

    for video_id in df['video_id'].unique():
        valid_frames = df[df['video_id'] == video_id]
        valid_landmarks = valid_frames[valid_frames['likelihood'] >= valid_threshold]

        if not valid_landmarks.empty:
            head_sizes = valid_landmarks.apply(lambda row: compute_head_size(row['landmarks']), axis=1)
            video_head_sizes[video_id] = np.median(head_sizes)

    return video_head_sizes

def normalize_coordinates(df: pd.DataFrame, valid_threshold: float = 0.5) -> pd.DataFrame:
    coordinate_cols = [col for col in df.columns if '_x' in col or '_y' in col]
    video_head_sizes = compute_median_head_size_per_video(df, valid_threshold)

    for video_id in df['video_id'].unique():
        norm_factor = 0.5 * (video_head_sizes.get(video_id) or 1)  # Avoid division by zero
        df.loc[df['video_id'] == video_id, coordinate_cols] /= norm_factor

    return df

def compute_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    df['mid_shoulder_x'] = (df['LeftShoulder_x'] + df['RightShoulder_x']) / 2.0
    df['mid_shoulder_y'] = (df['LeftShoulder_y'] + df['RightShoulder_y']) / 2.0
    df['head_rotation'] = np.sqrt((df['Chin_x'] - df['mid_shoulder_x'])**2 + (df['Chin_y'] - df['mid_shoulder_y'])**2)
    df['left_eye_chin'] = np.sqrt((df['LeftEye_x'] - df['Chin_x'])**2 + (df['LeftEye_y'] - df['Chin_y'])**2)
    df['right_eye_chin'] = np.sqrt((df['RightEye_x'] - df['Chin_x'])**2 + (df['RightEye_y'] - df['Chin_y'])**2)

    df['left_arm_x'] = df[['LeftShoulder_x', 'LeftElbow_x', 'LeftWrist_x']].mean(axis=1)
    df['left_arm_y'] = df[['LeftShoulder_y', 'LeftElbow_y', 'LeftWrist_y']].mean(axis=1)
    df['right_arm_x'] = df[['RightShoulder_x', 'RightElbow_x', 'RightWrist_x']].mean(axis=1)
    df['right_arm_y'] = df[['RightShoulder_y', 'RightElbow_y', 'RightWrist_y']].mean(axis=1)

    return df