import random
import time
import matplotlib.pyplot as plt

# -----------------------------
# Data Generation
# -----------------------------
def generate_data(n):
    return [random.randint(1, 100000) for _ in range(n)]

# -----------------------------
# Sorting Algorithms
# -----------------------------

# Bubble Sort (O(n^2))

# Merge Sort (O(n log n))
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Quick Sort (O(n log n) avg)
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# Built-in Sort (Timsort)
def python_sort(arr):
    return sorted(arr)

# -----------------------------
# Time Measurement
# -----------------------------
def measure_time(func, data):
    start = time.perf_counter()
    func(data)
    end = time.perf_counter()
    return end - start

# -----------------------------
# Main Experiment
# -----------------------------
sizes = [1000, 3000, 5000, 8000, 100000]

merge_times = []
quick_times = []
python_times = []

for size in sizes:
    print(f"\nRunning for size: {size}")
    
    data = generate_data(size)
    
    # Measure times
    merge_times.append(measure_time(merge_sort, data))
    quick_times.append(measure_time(quick_sort, data))
    python_times.append(measure_time(python_sort, data))

# -----------------------------
# Plot Graph
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(sizes, merge_times, label="Merge Sort (O(n log n))", marker='o')
plt.plot(sizes, quick_times, label="Quick Sort (avg O(n log n))", marker='o')
plt.plot(sizes, python_times, label="Built-in Sort (Timsort)", marker='o')

plt.xlabel("Input Size")
plt.ylabel("Time (seconds)")
plt.title("Sorting Algorithm Performance Comparison")
plt.legend()
plt.grid()

plt.show()

# -----------------------------
# Print Results
# -----------------------------
print("\n--- Results ---")
for i in range(len(sizes)):
    print(f"Size {sizes[i]}:")
    print(f"  Merge Sort : {merge_times[i]:.6f} sec")
    print(f"  Quick Sort : {quick_times[i]:.6f} sec")
    print(f"  Python Sort: {python_times[i]:.6f} sec")
