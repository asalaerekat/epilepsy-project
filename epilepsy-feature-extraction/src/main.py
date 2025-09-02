from src.config import Config
from src.data.io import load_data, save_data
from src.data.cleaning import clean_data
from src.data.splitting import split_data
from src.features.spatial import compute_spatial_features
from src.features.temporal import compute_temporal_derivatives
from src.features.aggregation import aggregate_features_by_interval
from src.features.motion import compute_motion_magnitude
from src.missingness.flagging import flag_missing_frames
from src.missingness.interpolation import interpolate_missing_values
from src.missingness.imputation import process_missing_data
from src.modeling.preprocessing import scale_numerical_features
from src.modeling.selection import select_features
from src.modeling.automl import train_model
from src.modeling.evaluation import evaluate_model

def main():
    # Load configuration
    config = Config()

    # Load data
    data = load_data(config.data_path)

    # Clean data
    cleaned_data = clean_data(data)

    # Flag missing frames
    flagged_data = flag_missing_frames(cleaned_data)

    # Split data into training and testing sets
    train_data, test_data = split_data(flagged_data)

    # Compute features
    spatial_features = compute_spatial_features(train_data)
    temporal_features = compute_temporal_derivatives(spatial_features)
    aggregated_features = aggregate_features_by_interval(temporal_features)
    motion_features = compute_motion_magnitude(aggregated_features)

    # Process missing data
    processed_data = process_missing_data(motion_features)

    # Scale numerical features
    scaled_data = scale_numerical_features(processed_data)

    # Select features
    selected_features = select_features(scaled_data)

    # Train model
    model = train_model(selected_features)

    # Evaluate model
    metrics = evaluate_model(model, test_data)

    # Save results
    save_data(metrics, config.output_path)

if __name__ == "__main__":
    main()