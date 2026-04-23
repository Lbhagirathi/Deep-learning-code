#!/usr/bin/env python3

import time
import cProfile
import timeit


# ---------------------------------------------------
# 1. Linear heavy computation
# Time Complexity: O(n)
# ---------------------------------------------------
def algo1(n):
    total = 0
    for i in range(n * 3000000):  # constant * n
        total += i
    return total


# ---------------------------------------------------
# 2. Nested loops
# Time Complexity: O(n^2)
# ---------------------------------------------------
def algo2(n):
    for i in range(n):
        for j in range(n):
            pass


# ---------------------------------------------------
# 3. Logarithmic loop
# Time Complexity: O(log n)
# ---------------------------------------------------
def algo3(n):
    i = 1
    while i < n:
        i *= 2


# ---------------------------------------------------
# 4. Mixed loop (n log n)
# Time Complexity: O(n log n)
# ---------------------------------------------------
def algo4(n):
    for i in range(n):
        j = 1
        while j < n:
            j *= 2


# ---------------------------------------------------
# 5. Exponential recursion
# T(n) = 2T(n-1)
# Time Complexity: O(2^n)
# ---------------------------------------------------
def algo5(n):
    if n <= 1:
        return
    algo5(n-1)
    algo5(n-1)


# ---------------------------------------------------
# 6. Divide recursion
# T(n) = 2T(n/2)
# Time Complexity: O(n)
# ---------------------------------------------------
def algo6(n):
    if n <= 1:
        return
    algo6(n//2)
    algo6(n//2)


# ---------------------------------------------------
# 7. Square root loop
# Time Complexity: O(sqrt(n))
# ---------------------------------------------------
def algo7(n):
    i = 1
    while i*i < n:
        i += 1


# ---------------------------------------------------
# 8. Triple nested loop
# Time Complexity: O(n^3)
# ---------------------------------------------------
def algo8(n):
    for i in range(n):
        for j in range(i):
            for k in range(j):
                pass


# ---------------------------------------------------
# 9. Recurrence: T(n) = T(n/2) + n
# Time Complexity: O(n)
# ---------------------------------------------------
def algo9(n):
    if n <= 1:
        return
    for i in range(n):
        pass
    algo9(n//2)


# ---------------------------------------------------
# 10. Doubling + inner loop
# 1 + 2 + 4 + ... + n = 2n
# Time Complexity: O(n)
# ---------------------------------------------------
def algo10(n):
    i = 1
    while i < n:
        for j in range(i):
            pass
        i *= 2


# ---------------------------------------------------
# 11. Profiling example
# study → O(n)
# relax → O(n)
# total → O(n)
# ---------------------------------------------------
def study(n):
    for i in range(n * 1000000):
        pass

def relax(n):
    time.sleep(0.1 * n)

def student(n):
    study(n)
    relax(n)
    study(n//2)


# ---------------------------------------------------
# MAIN
# Uncomment ANY function to test
# ---------------------------------------------------
if __name__ == "__main__":

    n = 1000

    print("Running algo1 (O(n))")
    algo1(n)

    # print("Running algo2 (O(n^2))")
    # algo2(n)

    # print("Running algo3 (O(log n))")
    # algo3(n)

    # print("Running algo4 (O(n log n))")
    # algo4(n)

    # print("Running algo5 (O(2^n))")
    # algo5(10)

    # print("Running algo6 (O(n))")
    # algo6(n)

    # print("Running algo7 (O(sqrt(n)))")
    # algo7(n)

    # print("Running algo8 (O(n^3))")
    # algo8(200)

    # print("Running algo9 (O(n))")
    # algo9(n)

    # print("Running algo10 (O(n))")
    # algo10(n)

    # print("Profiling student()")
    # cProfile.run("student(3)", sort="tottime")