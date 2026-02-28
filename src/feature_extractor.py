import numpy as np
from scipy.stats import skew

def calculate_sortedness(arr):
    count = 0
    for i in range(len(arr) - 1):
        if arr[i] <= arr[i + 1]:
            count += 1
    return count / (len(arr) - 1)

def calculate_duplicate_ratio(arr):
    return 1 - len(set(arr)) / len(arr)

def extract_features(arr):
    arr_np = np.array(arr)

    features = {
        "size": len(arr),
        "mean": np.mean(arr_np),
        "variance": np.var(arr_np),
        "range": np.max(arr_np) - np.min(arr_np),
        "sortedness": calculate_sortedness(arr),
        "duplicate_ratio": calculate_duplicate_ratio(arr),
        "skewness": skew(arr_np)
    }

    return list(features.values())