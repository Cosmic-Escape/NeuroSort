import joblib
import numpy as np
from src.feature_extractor import extract_features

MODEL_PATH = "models/neurosort_sota_regressor.pkl"
ALGO_ORDER = ["quick_sort", "merge_sort", "heap_sort", "insertion_sort"]

# Load SOTA regression model (log-transformed runtimes)
model = joblib.load(MODEL_PATH)


def predict_runtimes(data):
    """
    Predict runtimes for all algorithms using the trained SOTA regressor.
    Returns a dictionary mapping algorithm name -> predicted runtime in seconds.
    """

    # Extract features from dataset
    features = extract_features(data)

    # Predict log-transformed runtimes
    log_pred = model.predict([features])[0]

    # Inverse log-transform and clip to ensure positive runtimes
    predicted_times = np.exp(log_pred) - 1e-8
    predicted_times = np.clip(predicted_times, 0, None)

    # Map algorithm name -> predicted runtime
    runtime_dict = {algo: predicted_times[i] for i, algo in enumerate(ALGO_ORDER)}

    return runtime_dict


def predict_best_algorithm(data):
    """
    Predict the best sorting algorithm based on estimated runtimes.
    Returns:
        best_algo: str, name of algorithm with lowest predicted runtime
        runtime_dict: dict, predicted runtimes for all algorithms
    """
    runtime_dict = predict_runtimes(data)
    best_algo = min(runtime_dict, key=runtime_dict.get)
    return best_algo, runtime_dict