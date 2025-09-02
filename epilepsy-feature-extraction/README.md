# Epilepsy Feature Extraction

This project is designed for the extraction and analysis of features from epilepsy-related data. It provides a modular structure for handling data input/output, cleaning, feature extraction, missingness handling, modeling, and evaluation.

## Project Structure

The project is organized into the following directories and files:

- **src/**: Contains the main application code.
  - **main.py**: Entry point for the application.
  - **config.py**: Configuration settings for file paths and parameters.
  - **data/**: Handles data input/output and cleaning.
    - **io.py**: Functions for reading and writing data files.
    - **cleaning.py**: Functions for cleaning the data.
    - **splitting.py**: Functions for splitting the dataset into training and testing sets.
  - **features/**: Contains functions for feature extraction.
    - **spatial.py**: Functions for computing spatial features.
    - **temporal.py**: Functions for computing temporal features.
    - **aggregation.py**: Functions for aggregating features.
    - **motion.py**: Functions for computing motion-related features.
  - **missingness/**: Handles missing data.
    - **flagging.py**: Functions for flagging missing data.
    - **interpolation.py**: Functions for interpolating missing values.
    - **imputation.py**: Functions for imputing missing values.
  - **modeling/**: Contains functions for model training and evaluation.
    - **preprocessing.py**: Functions for data preprocessing.
    - **selection.py**: Functions for feature selection.
    - **automl.py**: Functions for automated machine learning.
    - **evaluation.py**: Functions for model evaluation.
  - **utils/**: Contains utility functions.
    - **helpers.py**: General utility functions used across modules.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To run the feature extraction process, execute the following command:

```
python src/main.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.