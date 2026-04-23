import time
import matplotlib.pyplot as plt

def algo(n):
    s = 0
    for i in range(n):
        for j in range(n):
            s += i*j
    return s

sizes = []
times = []

for n in range(100, 1000, 100):
    start = time.time()
    algo(n)
    end = time.time()

    sizes.append(n)
    times.append(end - start)

plt.plot(sizes, times)
plt.xlabel("Input size n")
plt.ylabel("Execution time")
plt.title("Time Complexity Plot")
plt.show()