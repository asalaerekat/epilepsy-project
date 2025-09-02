from sklearn.model_selection import train_test_split

def split_data_by_video(df, group_col='video_id', target_col='numeric_event', test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets while ensuring that all rows sharing the same
    group stay together, and the overall distribution of the target column is roughly maintained.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        group_col (str): Column name to group on (e.g., 'video_id').
        target_col (str): Column name of the target variable.
        test_size (float): Fraction of videos to include in the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_df (pd.DataFrame): DataFrame for training data.
        test_df (pd.DataFrame): DataFrame for testing data.
    """
    vid_to_label = df.groupby(group_col)[target_col].agg(lambda x: x.mode().iat[0])
    
    vids = vid_to_label.index.to_numpy()
    labels = vid_to_label.values

    train_vids, test_vids = train_test_split(
        vids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    train_df = df[df[group_col].isin(train_vids)].copy()
    test_df = df[df[group_col].isin(test_vids)].copy()

    overlap = set(train_df[group_col]).intersection(test_df[group_col])
    assert not overlap, f"Leakage detected: {overlap}"

    return train_df, test_df