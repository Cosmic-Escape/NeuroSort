import numpy as np
import random

def generate_dataset(size=1000, distribution="uniform", sortedness=0.0, duplicates_ratio=0.0):
    if distribution == "uniform":
        data = np.random.uniform(0, 1000, size)
    elif distribution == "normal":
        data = np.random.normal(500, 150, size)
    elif distribution == "exponential":
        data = np.random.exponential(300, size)
    else:
        raise ValueError("Unsupported distribution")

    data = data.tolist()

    # Introduce duplicates
    if duplicates_ratio > 0:
        num_duplicates = int(size * duplicates_ratio)
        for _ in range(num_duplicates):
            idx = random.randint(0, size - 1)
            data[idx] = data[random.randint(0, size - 1)]

    # Introduce partial sortedness
    if sortedness > 0:
        data.sort()
        swaps = int(size * (1 - sortedness))
        for _ in range(swaps):
            i, j = random.sample(range(size), 2)
            data[i], data[j] = data[j], data[i]

    return data