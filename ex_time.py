#!/usr/bin/env python3

import time
import random
import timeit

def student(n):
    if n < 1:
        raise Exception("In school, no time to sleep")
    elif n > 5:
        raise Exception("Making money, no time to sleep")
    else:
        time.sleep(n + random.uniform(-0.2, 0.2))

def exp1():
    start = time.time()
    student(2)
    end = time.time()

    print("Time taken:", end - start)

def exp2():
    times = []

    for _ in range(5):
        start = time.time()
        student(2)
        end = time.time()
        times.append(end - start)

    print("Runs:", times)
    print("Average:", sum(times)/len(times))

def exp3():
    times = 10
    t = timeit.timeit(lambda: student(2), number=times)
    print("Average (timeit):", t / times)

def exp4():
    times = 5
    for n in [1, 2, 3, 4, 5]:
        t = timeit.timeit(lambda: student(n), number=times)
        print(f"n={n}, time={t/times}")

if __name__ == "__main__":
    exp4()
