import fiona
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt
import matplotlib
matplotlib.use('TkAgg')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)

# Dosya yolu
file_path = "Ehyd/DLM_5000_BODENBEDECKUNG_20240507.gpkg"

# GeoPackage dosyas?ndaki tüm katmanlar? listele
with fiona.Env():
    layers = fiona.listlayers(file_path)
    print(layers)


# Belirli bir katman? açmak için
gdf_d = gpd.read_file(file_path, layer='DLM_AKTUALITAETSSTAND')
gdf_d.head()

gdf_b = gpd.read_file(file_path, layer='BOD_5300_WASSER_F')
gdf_b.head()
gdf_b.shape

# Geometry sütununu Shapely geometrisine çevirin
gdf_b['geometry'].head()

gdf_b_df = gpd.GeoDataFrame(gdf_b, geometry='geometry')

print(gdf_b_df.crs)

# CRS'yi EPSG:4326'ya dönü?tür
gdf_b_df = gdf_b_df.to_crs(epsg=4326)

# Görselle?tirme
fig, ax = plt.subplots(figsize=(10, 10))
gdf_b_df.plot(ax=ax, color='lightblue', edgecolor='blue')
ax.set_title('neyse ne', fontsize=15)
ax.set_axis_off()
plt.show()



######################################################################################

# Dosya yolu
file_path_akifer = "Ehyd\insp_ge_hg_aquifer_500k_epsg4258.gpkg"

# GeoPackage dosyas?ndaki tüm katmanlar? listele
with fiona.Env():
    layers = fiona.listlayers(file_path_akifer)
    print(layers)


# Belirli bir katman? açmak için
gdf_aqua = gpd.read_file(file_path_akifer, layer='aquiverview')
gdf_aqua.head()
gdf_aqua.shape

gdf_map = gpd.read_file(file_path_akifer, layer='mappedfeature')
gdf_map.head()
gdf_map.shape

gdf_aq = gpd.read_file(file_path_akifer, layer='aquifer')
gdf_aq.head()
gdf_aq.shape

gdf_comp = gpd.read_file(file_path_akifer, layer='compositionpart')
gdf_comp.head()
gdf_comp.shape

gdf_layer = gpd.read_file(file_path_akifer, layer='layer_styles')
gdf_layer.head()
gdf_layer.shape

# Geometry sütununu Shapely geometrisine çevirin
gdf_aqua['geometry'].head()

gdf_aqua_df = gpd.GeoDataFrame(gdf_aqua, geometry='geometry')

print(gdf_aqua_df.crs)

# CRS'yi EPSG:4326'ya dönü?tür
gdf_aqua_df = gdf_aqua_df.to_crs(epsg=4326)

# Görselle?tirme
fig, ax = plt.subplots(figsize=(12, 12))
gdf_aqua_df.plot(ax=ax, color='lightblue', edgecolor='blue')
ax.set_title('aqua', fontsize=15)
ax.set_axis_off()
plt.show()



# Geometry sütununu Shapely geometrisine çevirin
gdf_map['geometry'].head()

gdf_map_df = gpd.GeoDataFrame(gdf_map, geometry='geometry')

print(gdf_map_df.crs)

# CRS'yi EPSG:4326'ya dönü?tür
gdf_map_df = gdf_map_df.to_crs(epsg=4326)

# Görselle?tirme
fig, ax = plt.subplots(figsize=(12, 12))
gdf_map_df.plot(ax=ax, color='lightblue', edgecolor='blue')
ax.set_title('map', fontsize=15)
ax.set_axis_off()
plt.show()








import geopandas as gpd
from shapely.geometry import Point
from keplergl import KeplerGl

# Örne?in: Sadece gdf_aqua ve gdf_map geometrik veriler içeriyor olsun
# Bu verileri GeoDataFrame'e dönü?türme

# Tarih ve saat sütunlar?n? string format?na dönü?türme
for gdf in [gdf_aqua, gdf_map, gdf_aq, gdf_layer, gdf_comp]:
    for col in gdf.select_dtypes(include=['datetime64']).columns:
        gdf[col] = gdf[col].astype(str)

gdf_aqua_df = gpd.GeoDataFrame(gdf_aqua, geometry='geometry')
gdf_aqua_df = gdf_aqua_df.to_crs(epsg=4326)


gdf_map_df = gpd.GeoDataFrame(gdf_map, geometry='geometry')
gdf_map_df = gdf_map_df.to_crs(epsg=4326)



# KeplerGl haritas?n? olu?turma
map_1 = KeplerGl(height=600)

# Katmanlar? haritaya ekleme
map_1.add_data(data=gdf_aqua_df, name="aqua")
map_1.add_data(data=gdf_map_df, name="mapped")

# # Di?er geometrik olmayan katmanlar? oldu?u gibi ekleyin
# map_1.add_data(data=gdf_aq, name="akifer")
# map_1.add_data(data=gdf_layer, name="katman stili")
# map_1.add_data(data=gdf_comp, name="composition part")


# Haritay? kaydetme ve görüntüleme
map_1.save_to_html(file_name='keplergl_map.html')



