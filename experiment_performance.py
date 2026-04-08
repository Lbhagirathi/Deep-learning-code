import random

def generate_data(n):
    return [random.randint(1, 100000) for _ in range(n)]

import time

def measure_sort(sort_func, data):
    start = time.perf_counter()
    sort_func(data.copy())
    end = time.perf_counter()
    return end - start

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def python_sort(arr):
    arr.sort()

sizes = [1000, 5000, 10000, 20000]
results = []

for size in sizes:
    data = generate_data(size)
    
    t1 = measure_sort(python_sort, data)
    t2 = measure_sort(bubble_sort, data)
    
    results.append((size, t1, t2))

import matplotlib.pyplot as plt

sizes = [r[0] for r in results]
python_times = [r[1] for r in results]
bubble_times = [r[2] for r in results]

plt.plot(sizes, python_times, label="Built-in Sort")
plt.plot(sizes, bubble_times, label="Bubble Sort")

plt.xlabel("Input Size")
plt.ylabel("Time (seconds)")
plt.title("Sorting Algorithm Comparison")
plt.legend()
plt.show()
