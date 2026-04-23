import pandas as pd
import matplotlib.pyplot as plt

# Load life expectancy data
life_expectancy = pd.read_csv("data/life-expectancy.csv")  # no need for index_col=0
print (life_expectancy.columns)

coemissions = pd.read_csv("data/co-emissions-per-capita.csv")
print (coemissions.columns)

gdp = pd.read_csv("data/gdp-per-capita-maddison-project-database.csv")
print (gdp.columns)

trade=pd.read_csv("data/trade-as-share-of-gdp.csv")
print (trade.columns)

# Year to extract
year = 2020

# List of region files
region_files = [
    "regions/central-africa.txt",
    "regions/west-africa.txt",
    "regions/east-africa.txt",
    "regions/east-asia.txt",
    "regions/south-america.txt",
    "regions/southeast-asia.txt",
    "regions/south-asia.txt",
    "regions/europe-and-north-america.txt"
]

# Loop through each region
for region_file in region_files:
    region_name = region_file.split("/")[-1].replace(".txt","")
    
    # Read list of countries in this region
    with open(region_file, "r") as f:
        countries = [line.strip() for line in f.readlines()]
    
    # Filter life expectancy for these countries and the year
    df_life = life_expectancy[(life_expectancy['Entity'].isin(countries)) & (life_expectancy['Year'] == year)]
    df_co2 = coemissions[(coemissions['Entity'].isin(countries)) & (coemissions['Year'] == year)]
    df_gdp = gdp[(gdp['Entity'].isin(countries)) & (gdp['Year'] == year)]
    df_trade = trade[(trade['Entity'].isin(countries)) & (trade['Year'] == year)]
    
    # Save each region-indicator CSV
    df_life.to_csv(f"data/life_expectancy_{region_name}_{year}.csv", index=False)
    df_co2.to_csv(f"data/co2_per_capita_{region_name}_{year}.csv", index=False)
    df_gdp.to_csv(f"data/gdp_per_capita_{region_name}_{year}.csv", index=False)
    df_trade.to_csv(f"data/trade_share_gdp_{region_name}_{year}.csv", index=False)
    

    # df_region = df_life.merge(df_co2, on=['Entity','Year'], how='left')\
    #                 .merge(df_gdp, on=['Entity','Year'], how='left')\
    #                  .merge(df_trade, on=['Entity','Year'], how='left')
    
    # # Save to CSV
    # output_file = f"data/demographics_{region_name}_{year}.csv"
    # df_region.to_csv(output_file, index=False)
    # print(f"Saved merged indicators for {region_name}")


# Regions
regions = [
    "central-africa",
    "west-africa",
    "east-africa",
    "east-asia",
    "south-america",
    "southeast-asia",
    "south-asia",
    "europe-and-north-america"
]

value_cols = {
    "life": "Life expectancy",
    "co2": "CO₂ emissions per capita",
    "gdp": "GDP per capita",
    "trade": "Trade (% of GDP)"
}

# Define colors for regions
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']

plt.figure(figsize=(8,8))

for i, region in enumerate(regions):
    # Load each indicator CSV for the region
    life = pd.read_csv(f"data/life_expectancy_{region}_{year}.csv")[['Entity','Year',value_cols['life']]]
    co2 = pd.read_csv(f"data/co2_per_capita_{region}_{year}.csv")[['Entity','Year',value_cols['co2']]]
    gdp = pd.read_csv(f"data/gdp_per_capita_{region}_{year}.csv")[['Entity','Year',value_cols['gdp']]]
    trade = pd.read_csv(f"data/trade_share_gdp_{region}_{year}.csv")[['Entity','Year',value_cols['trade']]]
    
    # Merge datasets on Entity and Year
    df = life.merge(co2, on=['Entity','Year'], how='inner')\
             .merge(gdp, on=['Entity','Year'], how='inner')\
             .merge(trade, on=['Entity','Year'], how='inner')
    
    # Scatter plot
    plt.scatter(
        df['GDP per capita'],                # x-axis
        df['Life expectancy'],               # y-axis
        s=df['CO₂ emissions per capita']*20, # marker size
        c=df['Trade (% of GDP)'],            # marker color
        cmap='viridis',
        alpha=0.7,
        edgecolors='k',
        label=region
    )

plt.xlabel('GDP per capita')
plt.ylabel('Life expectancy')
plt.title('Demographics by Region for 2020')
plt.colorbar(label='Trade (% of GDP)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()