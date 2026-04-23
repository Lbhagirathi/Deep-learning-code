import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

# Argument parsing
parser = argparse.ArgumentParser(description="Plot life expectancy time series for a country")
parser.add_argument("file", help="CSV file containing life expectancy data for a country")
parser.add_argument("outdir", help="Directory where the plot will be saved")

args = parser.parse_args()

# Read data
df = pd.read_csv(args.file)

# Assuming column names are Country,Year,LifeExpectancy
year = df["Year"]
life = df["LifeExpectancy"]

# Extract country name from filename
country = os.path.basename(args.file).replace(".csv","")

# Plot
plt.figure()
plt.plot(year, life)
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.title(f"Life Expectancy in {country}")
plt.grid()

# Save plot
outfile = os.path.join(args.outdir, f"{country}.png")
plt.savefig(outfile)

plt.close()

# python plot_life_expectancy.py data-per-country/India.csv plots-per-country