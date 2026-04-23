import time
import matplotlib.pyplot as plt

def algorithm(n):
    # algorithm here
    pass

sizes = []
times = []

for n in range(100, 2000, 100):

    start = time.time()

    algorithm(n)

    end = time.time()

    sizes.append(n)
    times.append(end-start)

plt.plot(sizes, times)
plt.xlabel("n")
plt.ylabel("time")
plt.show()