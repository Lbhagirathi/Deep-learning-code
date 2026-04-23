import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('iip_wrt_prevyr.csv')

print(df.head())

#  df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
x_labels=df.iloc[:,0]
data=df.iloc[:,1:]

print(x_labels)
print(data)

x = np.arange(len(x_labels))

fig, ax=plt.subplots(figsize=(8, 6))

# bottom=np.zeros(len(df))
pos_bottom=np.zeros(len(df))
neg_bottom=np.zeros(len(df))
colors=[ 'LightSkyBlue', 'IndianRed', 'LightSalmon', 'Moccasin' ]

for i, column in enumerate(data.columns):
    values=data[column]
    pos_values=np.where(values>0,values,0)
    neg_values=np.where(values<0,values,0)

    ax.bar(x,pos_values,bottom=pos_bottom,color=colors[i],label=column)
    ax.bar(x,neg_values,bottom=neg_bottom,color=colors[i])
    pos_bottom+=pos_values
    neg_bottom+=neg_values
# for column in data.columns:
#     ax.bar(x, data[column], bottom=bottom, color=colors[column],label=column)
#     bottom+=data[column]



ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data', 0))


ax.text(0.01, 0.02,
    "Source: Index of Industrial Production",
    transform=ax.transAxes,
    color="lightgray",
    fontsize=9
)

ax.set_xlabel("Month", fontsize=12)
# ax.set_ylabel("IIP Growth (%)", fontsize=12)
ax.set_ylabel("Year-on-Year\nGrowth (%)", fontsize=12, labelpad=10)
ax.set_title("Index of Industrial Production", fontsize=14,pad=12)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=90)

ax.axhline(0, linestyle="--", linewidth=1, color="black")
# ax.axvline(0, linestyle="--", linewidth=1, color="black")

ax.legend(ncol=2)

plt.savefig("iip_plot.png", dpi=320)
plt.show()