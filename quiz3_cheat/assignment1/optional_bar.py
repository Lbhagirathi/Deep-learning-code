import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.arange(5)
y = [10, 15, 7, 12, 9]

# Example error values
y_error = [1, 2, 1.5, 1, 2]

fig, ax = plt.subplots(figsize=(8,5))

bars = ax.bar(
    x, y,

    width=0.6,                # width of bars
    bottom=0,                 # bottom edge position

    color="LightSkyBlue",     # bar fill color
    edgecolor="black",        # bar edge color
    linewidth=1.5,            # edge thickness

    yerr=y_error,             # vertical error bars
    xerr=None,                # horizontal error bars (optional)

    ecolor="red",             # error bar color
    capsize=6,                # length of error bar caps

    error_kw={"elinewidth":2}, # additional errorbar settings

    align="center",           # alignment: 'center' or 'edge'
    orientation="vertical",   # vertical bars
    log=False                 # log scale if True
)

# Axis labels and title
ax.set_xlabel("Category")
ax.set_ylabel("Values")
ax.set_title("Bar Chart Example with All Optional Arguments")

# Tick labels
ax.set_xticks(x)
ax.set_xticklabels(["A", "B", "C", "D", "E"])

plt.show()