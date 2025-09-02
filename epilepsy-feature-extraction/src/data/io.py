from pathlib import Path
import pandas as pd

def read_csv(file_path):
    """Reads a CSV file and returns a DataFrame."""
    return pd.read_csv(file_path)

def write_csv(dataframe, file_path):
    """Writes a DataFrame to a CSV file."""
    dataframe.to_csv(file_path, index=False)

def read_excel(file_path):
    """Reads an Excel file and returns a DataFrame."""
    return pd.read_excel(file_path)

def write_excel(dataframe, file_path):
    """Writes a DataFrame to an Excel file."""
    dataframe.to_excel(file_path, index=False)

def read_hdf(file_path):
    """Reads an HDF5 file and returns a DataFrame."""
    return pd.read_hdf(file_path)

def write_hdf(dataframe, file_path):
    """Writes a DataFrame to an HDF5 file."""
    dataframe.to_hdf(file_path, mode='w', format='table')