# ==========================================
# NETCDF CLIMATE MAP PLOTTING GUIDE
# (xarray + cartopy)
# Keep this as an exam reference
# ==========================================

import xarray as xr
import matplotlib.pyplot as pl
import cartopy.feature as cfeature
import cartopy.crs as ccrs

# ==========================================
# OPEN DATASET
# ==========================================

ds = xr.open_dataset('./20260406/assam_t2m_2025.nc')

print(ds)

# ==========================================
# SELECT VARIABLE
# ==========================================

t2m = ds['t2m']

# Convert Kelvin to Celsius
t2m = t2m - 273.15

# ==========================================
# TIME OPERATIONS (choose one)
# ==========================================

temp = t2m.mean(dim='valid_time')     # mean temperature
# temp = t2m.max(dim='valid_time')    # maximum temperature
# temp = t2m.min(dim='valid_time')    # minimum temperature
# temp = t2m[0,:,:]                   # first timestep
# temp = t2m.sel(valid_time='2025-01-01')  # specific date

# ==========================================
# CREATE FIGURE
# ==========================================

fig = pl.figure(figsize=(10,8))

# ==========================================
# MAP PROJECTIONS (choose one)
# ==========================================

projection = ccrs.PlateCarree()

# projection = ccrs.Mercator()
# projection = ccrs.Robinson()
# projection = ccrs.LambertConformal()
# projection = ccrs.PlateCarree(central_longitude=90)

ax = fig.add_axes([0.05,0.1,0.7,0.8], projection=projection)

# ==========================================
# COLORBAR AXIS
# ==========================================

cax = fig.add_axes([0.80,0.1,0.03,0.8])

# ==========================================
# DATA CRS
# ==========================================

data_crs = ccrs.PlateCarree()

# ==========================================
# PLOT METHODS (choose one)
# ==========================================

# --- PCOLORMESH (most common) ---
im = ax.pcolormesh(
    t2m['longitude'],
    t2m['latitude'],
    temp,
    cmap='bone',
    transform=data_crs
)

# --- CONTOURF ---
# im = ax.contourf(
#     t2m['longitude'],
#     t2m['latitude'],
#     temp,
#     levels=20,
#     cmap='coolwarm',
#     transform=data_crs
# )

# --- CONTOUR ---
# im = ax.contour(
#     t2m['longitude'],
#     t2m['latitude'],
#     temp,
#     colors='black',
#     transform=data_crs
# )

# ==========================================
# COLORMAP OPTIONS
# ==========================================

# bone
# coolwarm
# viridis
# jet
# plasma
# inferno
# RdBu
# Spectral

# ==========================================
# ADD MAP FEATURES
# ==========================================

ax.coastlines()

ax.add_feature(cfeature.BORDERS, color='black')

ax.add_feature(cfeature.RIVERS, linestyle='--')

# Optional features

# ax.add_feature(cfeature.LAKES)
# ax.add_feature(cfeature.OCEAN)
# ax.add_feature(cfeature.LAND)
# ax.add_feature(cfeature.STATES)

# ==========================================
# GRIDLINES
# ==========================================

gl = ax.gridlines(
    crs=data_crs,
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.5,
    linestyle='--'
)

# Optional gridline settings

# gl.top_labels = False
# gl.right_labels = False
# gl.xlabel_style = {'size':12}
# gl.ylabel_style = {'size':12}

# ==========================================
# ZOOM / EXTENT
# ==========================================

# ax.set_extent([88,97,24,29], crs=data_crs)  # Assam region

# ax.set_global()  # global map

# ==========================================
# TITLE
# ==========================================

pl.title("Mean 2m Temperature over Assam (2025)", fontsize=14)

# ==========================================
# COLORBAR
# ==========================================

cb = pl.colorbar(im, cax=cax)

cb.set_label("Temperature (°C)")

# ==========================================
# SAVE FIGURE
# ==========================================

# pl.savefig("assam_temperature_map.png", dpi=300)

# ==========================================
# SHOW PLOT
# ==========================================

pl.show()