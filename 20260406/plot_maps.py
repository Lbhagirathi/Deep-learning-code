import xarray as xr
import matplotlib.pyplot as pl
import cartopy.feature as cfeature
import cartopy.crs as ccrs

ds=xr.open_dataset('./20260406/assam_t2m_2025.nc')
print(ds)

t2m=ds['t2m']-273.15 #convert from k to deg c
print(t2m.values)

fig=pl.figure(figsize=[8,6])

crs=ccrs.PlateCarree()
ax=fig.add_axes([0.05,0.10,0.70,0.80],
                projection=ccrs.PlateCarree(central_longitude=90))

cax=fig.add_axes([0.85,0.10,0.05,0.80])

im=ax.pcolormesh(t2m['longitude'],t2m['latitude'],t2m.mean(dim='valid_time'),cmap=('bone'),transform=crs,) #sequential vs diverging colormap

crs=ccrs.PlateCarree()

ax.coastlines()
ax.add_feature(cfeature.BORDERS,color='k')
ax.add_feature(cfeature.RIVERS,color='k',linestyle='--')
gl=ax.gridlines(crs=crs,draw_labels=True,linewidth=0.5,color='gray',alpha=0.5,linestyle='-.')

pl.show()
