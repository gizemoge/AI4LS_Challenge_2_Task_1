import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)


# Yer alt? suyu verilerini yükleyin
groundwater = pd.read_csv("Ehyd/datasets/transformed_filtered_messstellen_gw.csv")
groundwater.head()
for col in groundwater.columns:
    print(col, groundwater[col].nunique())

# Rain
rain = pd.read_csv("Ehyd/datasets/transformed_nlv_rain.csv")
rain.head()

for col in rain.columns:
    print(col, rain[col].nunique())

rain['mpmua04'] = rain['mpmua04'].str.replace(' ', '').str.replace(',', '.').astype(float)


# Koordinatlar? seçin ve standartla?t?r?n
features_rain = rain[['longitude', 'latitude', 'mpmua04']].values
scaler = StandardScaler()
rain_scaled = scaler.fit_transform(features_rain)

#####
wcss = []  # WCSS: Within-Cluster Sum of Square

for i in range(1, 30):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(rain_scaled)
    wcss.append(kmeans.inertia_)

rain['cluster'] = kmeans.labels_

rain.head()
rain["cluster"].value_counts()

sns.scatterplot(x='longitude', y='latitude', hue='cluster', palette='cool', data=rain, s=50, marker='X', legend='full')

plt.title('Ya?mur Kümeleri')
plt.xlabel('Boylam')
plt.ylabel('Enlem')
plt.show()




# # KMeans modelini e?itin
# kmeans_rain = KMeans(n_clusters=5, random_state=42)  # Örne?in, 5 küme
# rain['cluster'] = kmeans_rain.fit_predict(coordinates_rain_scaled)


# Yer alt? suyu verilerini standartla?t?r?n
coordinates_groundwater = groundwater[['longitude', 'latitude']].values
coordinates_groundwater_scaled = scaler.transform(coordinates_groundwater)

# Yer alt? suyu verileri için kümeleri tahmin edin
groundwater['cluster'] = kmeans.predict(coordinates_groundwater_scaled)


# Haritay? çizin ve ya?mur kümeleri ile yer alt? suyu noktalar?n? gösterin
plt.figure(figsize=(16, 9))

# Ya?mur kümeleri
sns.scatterplot(x='longitude', y='latitude', hue='cluster', palette='viridis', data=rain, s=100, alpha=0.6, legend='full')

# Yer alt? suyu noktalar?
sns.scatterplot(x='longitude', y='latitude', hue='cluster', palette='cool', data=groundwater, s=50, marker='X', legend='full')

plt.title('Ya?mur Kümeleri ve Yer Alt? Suyu Noktalar?')
plt.xlabel('Boylam')
plt.ylabel('Enlem')
plt.show()







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




