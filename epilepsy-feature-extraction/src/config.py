# Configuration settings for the epilepsy feature extraction project

# File paths
DATA_PATH = '/Users/erekaa02/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Epilepsy Project/labeled_data_from_DLC/only_seizures_data.h5'
OUTPUT_PATH = '/Users/erekaa02/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Epilepsy Project/labeled_data_from_DLC'

# Parameters
FPS = 25  # Frames per second
INTERVAL_SECONDS = 1.0  # Duration of aggregation intervals in seconds
MISSINGNESS_THRESHOLD = 0.75  # Threshold for missing data
LIKELIHOOD_THRESHOLD = 0.6  # Threshold for likelihood to flag missing frames

# Feature extraction settings
FEATURES_TO_EXTRACT = [
    'spatial',
    'temporal',
    'motion',
    'aggregation'
]

# Model training settings
N_SPLITS = 5  # Number of splits for cross-validation
MAX_MODELS = 20  # Maximum number of models to train in AutoML
MAX_RUNTIME_SECS = 3600  # Maximum runtime for AutoML in seconds

# Logging settings
LOG_LEVEL = 'INFO'  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)