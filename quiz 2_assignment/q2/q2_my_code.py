import pandas as pd
import matplotlib.pyplot as pl

life_expectancy=pd.read_csv("data/life-expectancy.csv")
print(life_expectancy.columns)

gdp=pd.read_csv("data/gdp-per-capita-maddison-project-database.csv")
print(gdp.columns)

emissions=pd.read_csv("data/co-emissions-per-capita.csv")
print(emissions.columns)

trade=pd.read_csv("data/trade-as-share-of-gdp.csv")
print(trade.columns)

years=[1990, 2020]

regions=["regions/central-africa.txt",
         "regions/east-africa.txt",
         "regions/east-asia.txt",
         "regions/europe-and-north-america.txt",
         "regions/south-america.txt",
         "regions/south-asia.txt",
         "regions/southeast-asia.txt",
         "regions/west-africa.txt",
         ]

for year in years:
    for i in regions:
        region_name=i.split("/")[-1].replace(".txt","")

        with open(i,"r") as f:
            countries=[line.strip() for line in f.readlines()]

        df_life=life_expectancy[(life_expectancy['Entity'].isin(countries)) & (life_expectancy['Year']==year)]
        df_co2_emissions=emissions[(emissions['Entity'].isin(countries))&(emissions['Year']==year)]
        df_gdp=gdp[(gdp['Entity'].isin(countries)) & (gdp['Year']==year)]
        df_trade=trade[(trade['Entity'].isin(countries))&(trade['Year']==year)]


        df_life.to_csv(f"data/life_expectancy_{region_name}_{year}.csv",index=False)
        df_co2_emissions.to_csv(f"data/emissions_{region_name}_{year}.csv",index=False)
        df_gdp.to_csv(f"data/gdp_{region_name}_{year}.csv",index=False)
        df_trade.to_csv(f"data/trade_{region_name}_{year}.csv",index=False)

regions=["central-africa",
         "east-africa",
         "east-asia",
         "europe-and-north-america",
         "south-america",
         "south-asia",
         "southeast-asia",
         "west-africa"]

value_cols={
    "life":"Life expectancy",
    "gdp":"GDP per capita",
    "co2":"CO₂ emissions per capita",
    "trade":"Trade (% of GDP)"
}

colors=['violet','grey','blue','green','yellow','orange','red','pink']
region_colors = {region: colors[i] for i, region in enumerate(regions)}
marker_shapes = {1990:'^', 2020:'o'}  # triangles for 1990, circles for 2020

pl.figure(figsize=[8,8])

for year in years:
    for i, region in enumerate(regions):
        life=pd.read_csv(f"data/life_expectancy_{region}_{year}.csv")[['Entity','Year',value_cols['life']]]
        gdp=pd.read_csv(f"data/gdp_{region}_{year}.csv")[['Entity','Year',value_cols['gdp']]]
        co2=pd.read_csv(f"data/emissions_{region}_{year}.csv")[['Entity','Year',value_cols['co2']]]
        trade=pd.read_csv(f"data/trade_{region}_{year}.csv")[['Entity','Year',value_cols['trade']]]

        df=life.merge(co2,on=['Entity','Year'],how='inner')\
            .merge(gdp,on=['Entity','Year'],how='inner')\
            .merge(trade,on=['Entity','Year'],how='inner')
                
        # Scatter plot
        pl.scatter(
            df['GDP per capita'],
            df['Life expectancy'],
            s=(df['CO₂ emissions per capita'].fillna(0.1)*20)+10,
            c=region_colors[region],
            marker=marker_shapes[year],
            alpha=0.7,
            edgecolors='k'
        )
for region in regions:
    pl.scatter([], [], c=region_colors[region], alpha=0.7, s=50, edgecolors='k', label=region)

for year, marker in marker_shapes.items():
    pl.scatter([], [], c='k', alpha=0.7, s=50, marker=marker, label=str(year))

pl.xlabel('GDP per capita')
pl.ylabel('Life expectancy')
pl.title('Demographics by Region for 1990 and 2020')

pl.legend(title="Region & Year", bbox_to_anchor=(1.05,1), loc='upper left')
pl.tight_layout()
pl.show()