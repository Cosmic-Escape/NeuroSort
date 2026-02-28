import time
from src.sorting_algorithms import quick_sort, merge_sort, heap_sort, insertion_sort

ALGORITHMS = {
    "quick_sort": quick_sort,
    "merge_sort": merge_sort,
    "heap_sort": heap_sort,
    "insertion_sort": insertion_sort,
   
}

def benchmark_algorithms(arr, repeats=3):
    timings = {}

    for name, func in ALGORITHMS.items():
        total = 0
        for _ in range(repeats):
            start = time.perf_counter()
            func(arr.copy())
            end = time.perf_counter()
            total += (end - start)

        timings[name] = total / repeats

    best_algorithm = min(timings, key=timings.get)
    return best_algorithm, timings