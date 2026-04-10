"""
Loads all India monthly rainfall data and does some basic analysis
"""

import pandas as pd

def dataframe_descriptors(read):
    """
    Test basic dataframe descriptors in pandas
    """
    print("top rows of dataframe ...")    
    print(read.head())
    print("data frame info ...")
    print(read.info())
    print("summary statistics ...")
    print(read.describe())
    return None

if __name__=="__main__":
    print("running file...")
    print(__doc__)
    print(__file__)

    read=pd.read_csv("all-india-monthly-rainfall.csv", index_col=0)

    dataframe_descriptors(read)
    
    #use groupby
    read["Decade"]=(read.index//10)*10
    print(read["Decade"])
    decade_stats_jan=read.groupby('Decade')['Jan'].mean()
    print("Decadal mean for January ...")
    print(decade_stats_jan)
    summary_jan=read.groupby('Decade')['Jan'].agg(['min','max','mean'])

    print(summary_jan)

