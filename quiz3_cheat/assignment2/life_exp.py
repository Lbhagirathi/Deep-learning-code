import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("enteries_per_country.csv", header=None, names=["Country","Count"])

x = np.arange(len(df))

fig, ax = plt.subplots(figsize=(10,6))

ax.bar(x, df["Count"], color="LightSkyBlue")

# floating axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_position(('data',0))

# labels
ax.set_xlabel("Country")
ax.set_ylabel("Number of Entries")
ax.set_title("Entries per Country")

ax.set_xticks(x)
ax.set_xticklabels(df["Country"], rotation=90)

plt.tight_layout()
plt.show()