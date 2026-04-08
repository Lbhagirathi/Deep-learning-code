#!/usr/bin/env python3

import time
import random
import cProfile

def study(n):
    total = 0
    for i in range(n * 5000000):
        total += i * i
    return total

def relax(n):
    time.sleep(n * 0.2)

def student(n):
    if n < 1:
        raise Exception("In school, no time to sleep")
    elif n > 5:
        raise Exception("Making money, no time to sleep")
    else:
        study(n)
        relax(n)
        study(n // 2)

if __name__ == "__main__":
    cProfile.run("student(3)", sort="tottime")
