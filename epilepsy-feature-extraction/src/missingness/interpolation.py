def interpolate_missing_values(df, feature_cols):
    """
    Fills missing values using the average of the feature from the previous and next interval
    for the same video_id, when both exist. Otherwise, keeps the missing value.

    Args:
        df (pd.DataFrame): DataFrame with video_id, interval, and features.
        feature_cols (list): List of feature columns to interpolate.

    Returns:
        pd.DataFrame: DataFrame with interpolated values.
    """
    df_interpolated = df.copy()

    for feature in feature_cols:
        if feature in df_interpolated.columns:  # Ensure feature exists
            interpolated_series = (
                df_interpolated.groupby(['video_id', 'interval'])[feature]
                .apply(lambda group: group.interpolate(method='linear', limit_direction='both'))
                .reset_index(level=['video_id', 'interval'], drop=True)  # Fix MultiIndex issue
            )

            # Ensure the index is properly aligned before assignment
            df_interpolated.loc[interpolated_series.index, feature] = interpolated_series

    return df_interpolated