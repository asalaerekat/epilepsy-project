from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def one_hot_encode_categorical(df, categorical_cols):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_categorical = df[categorical_cols]
    transformed_data = encoder.fit_transform(df_categorical)
    encoded_columns = encoder.get_feature_names_out(categorical_cols)
    df_categorical_encoded = pd.DataFrame(transformed_data, columns=encoded_columns, index=df.index)
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, df_categorical_encoded], axis=1)
    return df

def scale_numerical_features(df, exclude_cols):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(exclude_cols).tolist()
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled