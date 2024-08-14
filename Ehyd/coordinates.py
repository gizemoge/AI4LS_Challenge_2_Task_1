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