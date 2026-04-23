#!/usr/bin/env python3

# ---------------------------------------------------
# Algorithm 1
# Triangular nested loops
# Time Complexity: O(n^2)
# ---------------------------------------------------
def algo1(n):
    count = 0
    for i in range(n):
        for j in range(i):
            count += 1
    return count


# ---------------------------------------------------
# Algorithm 2
# Linear loop with logarithmic inner loop
# Time Complexity: O(n log n)
# ---------------------------------------------------
def algo2(n):
    i = 1
    while i < n:
        j = 1
        while j < n:
            j *= 2
        i += 1


# ---------------------------------------------------
# Algorithm 3
# Nested logarithmic loops
# Time Complexity: O((log n)^2)
# ---------------------------------------------------
def algo3(n):
    i = 1
    while i < n:
        j = 1
        while j < i:
            j *= 2
        i *= 2


# ---------------------------------------------------
# Algorithm 4
# Recursive halving
# Recurrence: T(n) = 2T(n/2) + O(1)
# Time Complexity: O(n)
# ---------------------------------------------------
def algo4(n):
    if n <= 1:
        return
    algo4(n // 2)
    algo4(n // 2)


# ---------------------------------------------------
# Algorithm 5
# Unequal recursive split
# Recurrence: T(n) = T(n/2) + T(n/4)
# Time Complexity: O(n)
# ---------------------------------------------------
def algo5(n):
    if n <= 1:
        return
    algo5(n // 2)
    algo5(n // 4)


# ---------------------------------------------------
# Algorithm 6
# Triple nested loops
# Time Complexity: O(n^3)
# ---------------------------------------------------
def algo6(n):
    count = 0
    for i in range(n):
        for j in range(i):
            for k in range(j):
                count += 1
    return count


# ---------------------------------------------------
# Algorithm 7
# Square root complexity loop
# Condition: i*i < n
# Time Complexity: O(sqrt(n))
# ---------------------------------------------------
def algo7(n):
    i = 1
    while i * i < n:
        i += 1
    return i


# ---------------------------------------------------
# Algorithm 8
# Doubling loop
# Time Complexity: O(log n)
# ---------------------------------------------------
def algo8(n):
    i = 1
    while i < n:
        i *= 2
    return i


# ---------------------------------------------------
# MAIN FUNCTION
# Call any algorithm here
# ---------------------------------------------------
if __name__ == "__main__":

    n = 1000

    print("Running Algorithm 1:", algo1(n))
    # print("Running Algorithm 2"); algo2(n)
    # print("Running Algorithm 3"); algo3(n)
    # print("Running Algorithm 4"); algo4(n)
    # print("Running Algorithm 5"); algo5(n)
    # print("Running Algorithm 6:", algo6(n))
    # print("Running Algorithm 7:", algo7(n))
    # print("Running Algorithm 8:", algo8(n))