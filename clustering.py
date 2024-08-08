import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

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

