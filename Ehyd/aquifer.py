import pandas as pd
import geopandas as gpd
import fiona
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)

gpkg_path = 'Ehyd/insp_ge_hg_artificial_activewell_epsg4258.gpkg'

# List all layers in the GeoPackage
layers = fiona.listlayers(gpkg_path)
print(f"Layers in the GeoPackage: {layers}")

# Load a specific layer as a GeoDataFrame
gdf_activewell = gpd.read_file(gpkg_path, layer='activewell')
gdf_activewell.head()
gdf_activewell.shape # (2439, 15)

gdf_layerstyles = gpd.read_file(gpkg_path, layer='layer_styles')
gdf_layerstyles.head()
gdf_layerstyles.shape # (1, 12)

# Check the geometry type (e.g., Point, LineString, Polygon)
gdf_activewell.geometry.type.value_counts() # Point


###### https://ehao.boku.ac.at/#Grundlagen Hydrological atlas of austria

import geopandas as gpd
import matplotlib.pyplot as plt

# Load the shapefile (the other files are automatically read if they are in the same directory)
shapefile_path = "Ehyd/k1_3_wasserbilanz/k1_3_wasserbilanz.shp"


with fiona.Env():
    layers = fiona.listlayers(shapefile_path)
    print(layers)


wasser = gpd.read_file(shapefile_path, layer='k1_3_wasserbilanz')
wasser.head()

wasser.shape  # (5446, 39)

for col in wasser:
    print(wasser[col].nunique())

# Görselle?tirme
fig, ax = plt.subplots(figsize=(12, 12))
wasser.plot(ax=ax, color='lightblue', edgecolor='black')
ax.set_title('wasser', fontsize=15)
ax.set_axis_off()
plt.show()
