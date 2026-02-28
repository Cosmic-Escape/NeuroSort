import random
from src.dataset_generator import generate_dataset
from src.predictor import predict_best_algorithm
from src.benchmark import benchmark_algorithms

def main():
    print("\n=== NeuroSort CLI ===\n")

    size = int(input("Enter dataset size: "))
    distribution = input("Distribution (uniform/normal/exponential): ")

    data = generate_dataset(size=size, distribution=distribution)

    predicted = predict_best_algorithm(data)
    actual_best, timings = benchmark_algorithms(data)

    print("\nPredicted Best Algorithm:", predicted)
    print("Actual Best Algorithm:", actual_best)

    print("\nTiming Comparison:")
    for algo, time in timings.items():
        print(f"{algo}: {time:.6f} sec")

if __name__ == "__main__":
    main()