# File: /epilepsy-feature-extraction/epilepsy-feature-extraction/src/utils/helpers.py

def load_config(config_file):
    """
    Load configuration settings from a specified file.
    
    Args:
        config_file (str): Path to the configuration file.
    
    Returns:
        dict: Configuration settings as a dictionary.
    """
    import json
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def save_to_excel(dataframe, file_path):
    """
    Save a DataFrame to an Excel file.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        file_path (str): The path where the Excel file will be saved.
    """
    dataframe.to_excel(file_path, index=False)

def calculate_missing_percentage(df):
    """
    Calculate the percentage of missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
        pd.Series: A Series containing the percentage of missing values for each column.
    """
    return df.isnull().mean() * 100

def replace_missing_with_unknown(df, categorical_cols):
    """
    Replace missing values in specified categorical columns with 'Unknown'.
    
    Args:
        df (pd.DataFrame): The DataFrame to modify.
        categorical_cols (list): List of categorical columns to process.
    
    Returns:
        pd.DataFrame: The modified DataFrame with missing values replaced.
    """
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    return df

def map_labels(series, label_mapping):
    """
    Map values in a Series according to a specified mapping.
    
    Args:
        series (pd.Series): The Series to map.
        label_mapping (dict): A dictionary mapping old values to new values.
    
    Returns:
        pd.Series: The mapped Series.
    """
    return series.map(label_mapping)