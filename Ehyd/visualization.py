# coordinates


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pyproj import Transformer
import folium

pd.set_option('display.max_columns', None)

# Function to convert the coordinates to the desired format.
def transform_coordinates(df):
    # Finding column names starting with x and y
    x_cols = [col for col in df.columns if col.startswith('x')]
    y_cols = [col for col in df.columns if col.startswith('y')]

    # Converting columns to string, replacing comma with dot, and converting to float
    for col in x_cols + y_cols:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    # Defining the transformer
    transformer = Transformer.from_crs("EPSG:31287", "EPSG:4326", always_xy=True)

    # Adding transformed coordinates
    for x_col, y_col in zip(x_cols, y_cols):
        coords = df.apply(lambda row: transformer.transform(row[x_col], row[y_col]), axis=1)
        df['longitude'] = coords.apply(lambda coord: coord[0])
        df['latitude'] = coords.apply(lambda coord: coord[1])

    return df

"""
# Visualization of the Groundwater data with the Agglomerative Clustering.
X = filtered_messstellen_gw[['latitude', 'longitude']]

agg_clustering = AgglomerativeClustering(n_clusters=100)
clusters = agg_clustering.fit_predict(X)

filtered_messstellen_gw['cluster'] = clusters

# Scatter plot
plt.figure(figsize=(19, 10))
unique_clusters = np.unique(clusters)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_clusters)))

# Scatter plot for each cluster
for i, cluster in enumerate(unique_clusters):
    cluster_points = X[clusters == cluster]
    plt.scatter(cluster_points['longitude'], cluster_points['latitude'],
                label=f'Cluster {cluster}', s=50, c=[colors[i]], marker='o')


plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Agglomerative Clustering of Locations')
plt.legend(title="Clusters", loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()  # Legend'?n grafikle daha iyi yerle?mesini sa?lar
plt.show()


# Map visualization of the Sources data.
locations = messstellen_qu[['latitude', 'longitude']]

m = folium.Map(location=[locations['latitude'].mean(), locations['longitude'].mean()], zoom_start=8)

for _, row in locations.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Lat: {row['latitude']}, Lon: {row['longitude']}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

m.save('austria_qu_deneme.html')


# Map visualization of the Surface Water data.
locations = messstellen_owf[['latitude', 'longitude']]

m = folium.Map(location=[locations['latitude'].mean(), locations['longitude'].mean()], zoom_start=8)

for _, row in locations.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Lat: {row['latitude']}, Lon: {row['longitude']}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

m.save('austria_owf.html')
"""


# burada filtrelenmemi? hallerini de koordinatlar? ekleyece?im:
filtered_messstellen_gw = pd.read_csv("ehyd/datasets/filtered_messstellen_gw.csv", sep=',', encoding='windows-1252')
filtered_messstellen_nvl = pd.read_csv("ehyd/datasets/filtered_messstellen_nvl.csv", sep=';', encoding='windows-1252')

messstellen_gw = pd.read_csv("ehyd/datasets/messstellen_gw.csv", sep=';', encoding='windows-1252')
messstellen_nlv = pd.read_csv("ehyd/datasets/messstellen_nlv.csv", sep=';', encoding='windows-1252')
messstellen_owf = pd.read_csv("ehyd/datasets/messstellen_owf.csv", sep=';', encoding='windows-1252')
messstellen_qu = pd.read_csv("ehyd/datasets/messstellen_qu.csv", sep=';', encoding='windows-1252')

transformed_filtered_messstellen_gw = transform_coordinates(filtered_messstellen_gw)
transformed_messstellen_gw = transform_coordinates(messstellen_gw)
transformed_messstellen_nlv = transform_coordinates(messstellen_nlv)
transformed_messstellen_owf = transform_coordinates(messstellen_owf)
transformed_messstellen_qu = transform_coordinates(messstellen_qu)

# kay?tl? çok az csv varm?? gibi bi de bunu kaydettim:
transformed_filtered_messstellen_gw.to_csv('ehyd/datasets/transformed_filtered_messstellen_gw.csv', index=False)
transformed_messstellen_gw.to_csv('ehyd/datasets/transformed_messstellen_gw.csv', sep=';', index=False)
transformed_messstellen_nlv.to_csv('ehyd/datasets/transformed_messstellen_nlv.csv', sep=';', index=False)
transformed_messstellen_owf.to_csv('ehyd/datasets/transformed_messstellen_owf.csv', sep=';', index=False)
transformed_messstellen_qu.to_csv('ehyd/datasets/transformed_messstellen_qu.csv', sep=';', index=False)

# bu dosyada koordinatlar? standartla?t?r?p veri setlerine eklemi? oldum

transformed_messstellen_nlv.head()
transformed_messstellen_owf.head()
transformed_messstellen_qu.head()
transformed_messstellen_gw.head()
transformed_filtered_messstellen_gw.head()

####################################################################
# deneme

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



###################################################################
# visualization
# imports
import pandas as pd
import folium
from pyproj import Transformer
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from shapely.geometry import Point

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')

############################################################################################################
# ?stenen tüm lokasyonlar? haritada görelim:

# Veriyi yükle
filtered_messstellen_gw = pd.read_csv("datasets/filtered_messstellen_gw.csv", sep=',', encoding='windows-1252')

# Koordinat sütunlar?n? do?ru formata dönü?tür
coordinates_of_locations = filtered_messstellen_gw[['hzbnr01', 'xrkko09', 'yhkko10']]
coordinates_of_locations['xrkko09'] = coordinates_of_locations['xrkko09'].astype(str)
coordinates_of_locations['yhkko10'] = coordinates_of_locations['yhkko10'].astype(str)
coordinates_of_locations['xrkko09'] = coordinates_of_locations['xrkko09'].str.replace(',', '.').astype(float)
coordinates_of_locations['yhkko10'] = coordinates_of_locations['yhkko10'].str.replace(',', '.').astype(float)

coordinates_of_locations.head()

# Create a transformer object for coordinate conversion
transformer = Transformer.from_crs("EPSG:31287", "EPSG:4326", always_xy=True)

# Function to convert coordinates
def convert_coords(row):
    lon, lat = transformer.transform(row['xrkko09'], row['yhkko10'])
    return pd.Series({'latitude': lat, 'longitude': lon})

# Apply the conversion to each row
coordinates_of_locations[['latitude', 'longitude']] = coordinates_of_locations.apply(convert_coords, axis=1)

# Create a map centered on Austria
m = folium.Map(location=[47.5162, 14.5501], zoom_start=7)

# Add markers for each location
for idx, row in coordinates_of_locations.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Basin Code: {row['hzbnr01']}",
        tooltip=f"Basin Code: {row['hzbnr01']}").add_to(m)

# Save the map
m.save("austria_water_basins_map.html")

print("Map has been saved as 'austria_water_basins_map.html'. Open this file in a web browser to view the map.")

##################################################################################################################333

messstellen_nvl = pd.read_csv("datasets/messstellen_nlv.csv", sep=';', encoding='windows-1252')

# Koordinat sütunlar?n? do?ru formata dönü?tür
coordinates_of_nvl = messstellen_nvl[['hzbnr01', 'xrkko08', 'yhkko09']]
coordinates_of_nvl['xrkko08'] = coordinates_of_nvl['xrkko08'].astype(str)
coordinates_of_nvl['yhkko09'] = coordinates_of_nvl['yhkko09'].astype(str)
coordinates_of_nvl['xrkko08'] = coordinates_of_nvl['xrkko08'].str.replace(',', '.').astype(float)
coordinates_of_nvl['yhkko09'] = coordinates_of_nvl['yhkko09'].str.replace(',', '.').astype(float)


def convert_coords(row):
    lon, lat = transformer.transform(row['xrkko08'], row['yhkko09'])
    return pd.Series({'latitude': lat, 'longitude': lon})
coordinates_of_nvl[['latitude', 'longitude']] = coordinates_of_nvl.apply(convert_coords, axis=1)

# Create a map centered on Austria
m = folium.Map(location=[47.5162, 14.5501], zoom_start=7)

# Add markers for each location
for idx, row in coordinates_of_nvl.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Basin Code: {row['hzbnr01']}",
        tooltip=f"Basin Code: {row['hzbnr01']}").add_to(m)

# Save the map
m.save("austria_nvl.html")

print("Map has been saved as 'austria_nvl.html'. Open this file in a web browser to view the map.")

# bu bize ?unu gösterdi ki neredeyse her noktaya ait ya??? verileri var.hjhgku

################################################################################################
# map
# bunun ad? eskiden clustring di ?imdi map yapt?k.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np
pd.set_option('display.max_columns', None)

df = pd.read_csv("datasets/transformed_filtered_messstellen_gw.csv")
df.head()

df["code04"].nunique()  # 54
df["code03"].nunique()  # 61
df["mstnam02"].nunique()  # 487
df["hzbnr01"].nunique()  # 487
df["dbmsnr"].nunique()  # 487
df["gwgeb03"].nunique()  # 61
df["gwkoerpe04"].nunique()  # 54

df.info()
df.isnull().sum()

df.head()

# ?lgili sütunlar? seçme
features2 = df[['longitude', 'latitude', "gwgeb03", "gwkoerpe04"]]
features = df[['longitude', 'latitude', "gokmua05", "gwtmug06"]]

# Eksik de?erleri kontrol etme ve doldurma
features['gokmua05'] = features['gokmua05'].str.replace(' ', '').str.replace(',', '.').astype(float)
features['gwtmug06'] = features['gwtmug06'].str.replace(' ', '').str.replace(',', '.').astype(float)

features["gokmua05-gwtmug06"] = features['gokmua05'] - features['gwtmug06']


scaler = StandardScaler()
features["gwgeb03", "gwkoerpe04", "gokmua05-gwtmug06"] = scaler.fit_transform(features["gwgeb03", "gwkoerpe04", "gokmua05-gwtmug06"])

wcss = []  # WCSS: Within-Cluster Sum of Square

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

df['cluster'] = kmeans.labels_

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Optimum küme say?s?n? belirleyin (örne?in 3)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(scaled_features)

# Sonuçlar? DataFrame'e ekleme
df['Cluster'] = clusters


plt.figure(figsize=(10, 7))
sns.scatterplot(x=df['longitude'], y=df['latitude'], hue=df['Cluster'], palette='viridis')
plt.title('Clustering Results')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()



# Kategorik sütunlar? say?sal de?erlere dönü?türme
df['gokmua05'] = df['gokmua05'].str.replace(',', '.').astype(float)
df['gwtmug06'] = df['gwtmug06'].str.replace(',', '.').astype(float)

# Renk paletini olu?tur
unique_codes = df['gwgeb03'].unique()
palette = sns.color_palette("hsv", len(unique_codes))
color_dict = dict(zip(unique_codes, palette))

# Haritay? çizme
plt.figure(figsize=(16, 9))
sns.scatterplot(x='longitude', y='latitude', hue='gwgeb03', palette=color_dict, data=df, s=100, legend=False)

plt.title('Koordinatlar ve gwkoerpe04 Kodu')
plt.xlabel('Boylam')
plt.ylabel('Enlem')
plt.show()



# Rain
rain = pd.read_csv("datasets/transformed_nlv_rain.csv")
rain.head()

# Tüm sütunlardaki benzersiz de?erleri bulma
unique_values = {col: rain[col].nunique() for col in rain.columns}

for col, values in unique_values.items():
    print(f"Sütun {col} için benzersiz de?erler:\n{values}\n")

# Renk paletini olu?tur
unique_rain = rain['gew03'].unique()
palette = sns.color_palette("hsv", len(unique_rain))
color_dict_rain = dict(zip(unique_rain, palette))

# Haritay? çizme
plt.figure(figsize=(16, 9))
sns.scatterplot(x='longitude', y='latitude', hue='gew03', palette=color_dict_rain, data=rain, s=100, legend=False)

plt.title('Koordinatlar ve gew03 Kodu')
plt.xlabel('Boylam')
plt.ylabel('Enlem')
plt.show()


# Snow
snow = pd.read_csv("datasets/transformed_nlv_snow.csv")
snow.head()

# Tüm sütunlardaki benzersiz de?erleri bulma
unique_values = {col: snow[col].nunique() for col in snow.columns}

for col, values in unique_values.items():
    print(f"Sütun {col} için benzersiz de?erler:\n{values}\n")

# Renk paletini olu?tur
unique_snow = snow['gew03'].unique()
palette = sns.color_palette("hsv", len(unique_snow))
color_dict_snow = dict(zip(unique_snow, palette))


# Haritay? çizme
plt.figure(figsize=(16, 9))
sns.scatterplot(x='longitude', y='latitude', hue='gew03', palette=color_dict_snow, data=snow, s=100, legend=False)
plt.title('Koordinatlar ve gew03 Kodu')
plt.xlabel('Boylam')
plt.ylabel('Enlem')
plt.show()

#####################################################################################################
# akifer
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
