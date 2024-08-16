import pandas as pd
import geopandas as gpd
import fiona

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
