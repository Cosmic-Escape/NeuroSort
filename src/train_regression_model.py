import random
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# SOTA regressors
import xgboost as xgb
import lightgbm as lgb

from src.dataset_generator import generate_dataset
from src.feature_extractor import extract_features
from src.benchmark import benchmark_algorithms

ALGO_ORDER = ["quick_sort", "merge_sort", "heap_sort", "insertion_sort"]


def generate_training_data(samples=1200):
    X = []
    y = []

    for _ in range(samples):

        size = random.choice([
            random.randint(10, 50),
            random.randint(200, 1000),
            random.randint(2000, 5000)
        ])

        distribution = random.choice(["uniform", "normal", "exponential"])
        sortedness = random.uniform(0, 1)
        duplicates_ratio = random.uniform(0, 0.6)

        data = generate_dataset(
            size=size,
            distribution=distribution,
            sortedness=sortedness,
            duplicates_ratio=duplicates_ratio
        )

        features = extract_features(data)
        _, timings = benchmark_algorithms(data)

        # Log-transform runtimes for stability and positive constraints
        runtime_vector = [np.log(timings[algo] + 1e-8) for algo in ALGO_ORDER]

        X.append(features)
        y.append(runtime_vector)

    return pd.DataFrame(X), np.array(y)


def train_sota_model():
    print("Generating regression training data...")
    X, y = generate_training_data(samples=1200)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Build a SOTA regression pipeline
    # -------------------------------
    base_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        verbosity=0
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # scale features for better convergence
        ("regressor", MultiOutputRegressor(base_model))
    ])

    print("Training SOTA regression model...")
    pipeline.fit(X_train, y_train)

    # Predict and exponentiate to get back runtime
    y_pred = np.exp(pipeline.predict(X_test)) - 1e-8
    y_true = np.exp(y_test) - 1e-8

    mae = mean_absolute_error(y_true, y_pred)
    print(f"\nMean Absolute Error (runtime in sec): {mae:.8f}")

    joblib.dump(pipeline, "models/neurosort_sota_regressor.pkl")
    print("\nSOTA Regression model saved.")


if __name__ == "__main__":
    train_sota_model()