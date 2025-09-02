from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def interpolate_missing_values(df, feature_cols):
    df_interpolated = df.copy()

    for feature in feature_cols:
        if feature in df_interpolated.columns:
            interpolated_series = (
                df_interpolated.groupby(['video_id', 'interval'])[feature]
                .apply(lambda group: group.interpolate(method='linear', limit_direction='both'))
                .reset_index(level=['video_id', 'interval'], drop=True)
            )

            df_interpolated.loc[interpolated_series.index, feature] = interpolated_series

    return df_interpolated

def process_missing_data_with_interpolation_and_imputation(df, threshold=0.5, max_iter=10):
    feature_cols = [col for col in df.columns if col not in ['video_id', 'interval', 'event', 'segment', 'left_arm_state', 'right_arm_state']]

    df['missing_ratio'] = df[feature_cols].isna().mean(axis=1)
    df_cleaned = df[df['missing_ratio'] <= threshold]

    df_cleaned = interpolate_missing_values(df_cleaned, feature_cols)

    imputer = IterativeImputer(max_iter=max_iter, random_state=42)
    df_cleaned[feature_cols] = imputer.fit_transform(df_cleaned[feature_cols])

    return df_cleaned