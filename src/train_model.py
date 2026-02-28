import random
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.dataset_generator import generate_dataset
from src.feature_extractor import extract_features
from src.benchmark import benchmark_algorithms


def generate_training_data(samples=1000):
    X = []
    y = []

    for _ in range(samples):

        # Force diversity
        size = random.choice([
            random.randint(10, 50),        # small → insertion
            random.randint(200, 1000),     # medium
            random.randint(2000, 5000)     # large
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
        best_algo, timings = benchmark_algorithms(data)

        X.append(features)
        y.append(best_algo)

    return pd.DataFrame(X), y


def train_model():
    print("Generating training data...")
    X, y = generate_training_data(samples=700)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        random_state=42
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("\nModel Evaluation:")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    joblib.dump(model, "models/neurosort_model.pkl")
    print("\nModel saved to models/neurosort_model.pkl")


if __name__ == "__main__":
    train_model()