# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# (1) LOAD DATA
# -----------------------------

# File paths
trade_file = "/nfscommon/common/bedartha/quiz2/trade-as-share-of-gdp.csv"
co2_file = "/nfscommon/common/bedartha/quiz2/co-emissions-per-capita.csv"

# Read CSV files
trade_df = pd.read_csv(trade_file)
co2_df = pd.read_csv(co2_file)

# -----------------------------
# (2) FILTER DATA (India & China, 1960–2024)
# -----------------------------

countries = ["India", "China"]

trade_df = trade_df[
    (trade_df["Entity"].isin(countries)) &
    (trade_df["Year"] >= 1960) &
    (trade_df["Year"] <= 2024)
]

co2_df = co2_df[
    (co2_df["Entity"].isin(countries)) &
    (co2_df["Year"] >= 1960) &
    (co2_df["Year"] <= 2024)
]

# Pivot data for easier plotting
trade_pivot = trade_df.pivot(index="Year", columns="Entity", values=trade_df.columns[-1])
co2_pivot = co2_df.pivot(index="Year", columns="Entity", values=co2_df.columns[-1])

# -----------------------------
# (3) CREATE FIGURE LAYOUT
# -----------------------------

# Create figure of size 8x6 inches
fig = plt.figure(figsize=(8, 6))

# Grid layout: 2 rows, 2 columns
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)  # top wide plot
ax2 = ax1.twinx()  # twin axis

ax3 = plt.subplot2grid((2, 2), (1, 0))  # bottom left
ax4 = plt.subplot2grid((2, 2), (1, 1))  # bottom right

# -----------------------------
# (4) PLOT TIME SERIES (ax1 & ax2)
# -----------------------------

# Colors for countries
colors = {"India": "blue", "China": "red"}

# Line styles for data types
linestyle_trade = "-"
linestyle_co2 = "--"

# Plot trade data (ax1)
for country in countries:
    ax1.plot(trade_pivot.index,
             trade_pivot[country],
             color=colors[country],
             linestyle=linestyle_trade,
             label=f"{country} Trade")

# Plot CO2 emissions (ax2)
for country in countries:
    ax2.plot(co2_pivot.index,
             co2_pivot[country],
             color=colors[country],
             linestyle=linestyle_co2,
             label=f"{country} CO2")

# -----------------------------
# (5) LEGENDS (top-left & top-right)
# -----------------------------

ax1.legend(loc="upper left")   # trade legend
ax2.legend(loc="upper right")  # CO2 legend

# -----------------------------
# (6) SCATTER PLOTS (ax3 & ax4)
# -----------------------------

# Merge datasets on Year for scatter plotting
merged = pd.merge(trade_pivot, co2_pivot, left_index=True, right_index=True,
                  suffixes=("_trade", "_co2"))

years = merged.index

# ax3: Trade India vs China
sc1 = ax3.scatter(
    merged["India_trade"],
    merged["China_trade"],
    c=years,
    cmap="viridis"
)

# ax4: CO2 India vs China
sc2 = ax4.scatter(
    merged["India_co2"],
    merged["China_co2"],
    c=years,
    cmap="plasma"
)

# Add colorbars
cbar1 = plt.colorbar(sc1, ax=ax3)
cbar1.set_label("Year")

cbar2 = plt.colorbar(sc2, ax=ax4)
cbar2.set_label("Year")

# -----------------------------
# (7) AXIS LABELS
# -----------------------------

# Top plot
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Trade (% of GDP)", fontsize=12)
ax2.set_ylabel("CO2 Emissions per Capita", fontsize=12)

# Bottom plots
ax3.set_xlabel("India Trade", fontsize=12)
ax3.set_ylabel("China Trade", fontsize=12)

ax4.set_xlabel("India CO2", fontsize=12)
ax4.set_ylabel("China CO2", fontsize=12)

# -----------------------------
# (8) TICK LABEL SIZE
# -----------------------------

for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(axis='both', labelsize=10)

# -----------------------------
# (9) FINAL TOUCHES
# -----------------------------

plt.tight_layout()
plt.show()