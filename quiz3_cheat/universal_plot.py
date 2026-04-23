#!/usr/bin/env python3

import time
import matplotlib.pyplot as plt


# ---------------------------------------------------
# Algorithms
# ---------------------------------------------------

# O(n^2)
def algo1(n):
    count = 0
    for i in range(n):
        for j in range(i):
            count += 1
    return count


# O(n log n)
def algo2(n):
    i = 1
    while i < n:
        j = 1
        while j < n:
            j *= 2
        i += 1


# O((log n)^2)
def algo3(n):
    i = 1
    while i < n:
        j = 1
        while j < i:
            j *= 2
        i *= 2


# O(n)
def algo4(n):
    if n <= 1:
        return
    algo4(n // 2)
    algo4(n // 2)


# O(n)
def algo5(n):
    if n <= 1:
        return
    algo5(n // 2)
    algo5(n // 4)


# O(n^3)
def algo6(n):
    count = 0
    for i in range(n):
        for j in range(i):
            for k in range(j):
                count += 1
    return count


# O(sqrt(n))
def algo7(n):
    i = 1
    while i * i < n:
        i += 1
    return i


# O(log n)
def algo8(n):
    i = 1
    while i < n:
        i *= 2
    return i


# ---------------------------------------------------
# PLOTTING FUNCTION (THIS IS WHAT YOU NEED)
# ---------------------------------------------------

def plot_algorithm(algo_func, name, start=100, end=2000, step=100):

    sizes = []
    times = []

    for n in range(start, end, step):

        start_time = time.time()
        algo_func(n)
        end_time = time.time()

        sizes.append(n)
        times.append(end_time - start_time)

    plt.plot(sizes, times)
    plt.xlabel("Input size (n)")
    plt.ylabel("Execution time")
    plt.title(f"Time Complexity of {name}")
    plt.show()


# ---------------------------------------------------
# MAIN (CALL ANY ALGORITHM HERE)
# ---------------------------------------------------

if __name__ == "__main__":

    # Just change this line to test anything

    plot_algorithm(algo1, "Algo1 O(n^2)")
    # plot_algorithm(algo2, "Algo2 O(n log n)")
    # plot_algorithm(algo3, "Algo3 O((log n)^2)")
    # plot_algorithm(algo4, "Algo4 O(n)")
    # plot_algorithm(algo5, "Algo5 O(n)")
    # plot_algorithm(algo6, "Algo6 O(n^3)", start=10, end=200, step=10)
    # plot_algorithm(algo7, "Algo7 O(sqrt(n))")
    # plot_algorithm(algo8, "Algo8 O(log n)")