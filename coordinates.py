import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pyproj import Transformer

# messstellen csv lerdeki bütün koordinatlar? düzelyelim:

filtered_messstellen_nvl = pd.read_csv("datasets/filtered_messstellen_nvl.csv", sep=';', encoding='windows-1252')
filtered_messstellen_gw = pd.read_csv("datasets/filtered_messstellen_gw.csv", sep=',', encoding='windows-1252')

filtered_messstellen_gw['xrkko09'] = filtered_messstellen_gw['xrkko09'].astype(str).str.replace(',', '.').astype(float)
filtered_messstellen_gw['yhkko10'] = filtered_messstellen_gw['yhkko10'].astype(str).str.replace(',', '.').astype(float)

filtered_messstellen_nvl['xrkko08'] = filtered_messstellen_nvl['xrkko08'].astype(str).str.replace(',', '.').astype(float)
filtered_messstellen_nvl['yhkko09'] = filtered_messstellen_nvl['yhkko09'].astype(str).str.replace(',', '.').astype(float)

filtered_messstellen_nvl.head()


# Koordinat dönü?ümünü gerçekle?tirecek transformer objesini tan?mlay?n
transformer = Transformer.from_crs("EPSG:31287", "EPSG:4326", always_xy=True)  # Burada 'xyz' ve 'abc' uygun EPSG kodlar? ile de?i?tirilmelidir

def convert_coords(x, y):
    """
    x ve y koordinatlar?n? dönü?türerek enlem ve boylam de?erlerini döndürür.

    Args:
        x (float): X koordinat?.
        y (float): Y koordinat?.

    Returns:
        pd.Series: Enlem ve boylam de?erlerini içeren bir Pandas Series nesnesi.
    """
    lon, lat = transformer.transform(x, y)
    return pd.Series({'latitude': lat, 'longitude': lon})





filtered_messstellen_nvl[['latitude', 'longitude']] = filtered_messstellen_nvl.apply(lambda row: convert_coords(row['xrkko08'], row['yhkko09']), axis=1)

filtered_messstellen_nvl.to_csv('datasets/filtered_messstellen_nvl.csv', index=False, encoding='windows-1252')
filtered_messstellen_gw.to_csv('datasets/filtered_messstellen_gw.csv', index=False, encoding='windows-1252')





# Target
filtered_messstellen_gw = pd.read_csv("datasets/filtered_messstellen_gw.csv", sep=',', encoding='windows-1252')

X = filtered_messstellen_gw[['latitude', 'longitude']]

# Agglomerative Clustering modelini tan?mlay?n
agg_clustering = AgglomerativeClustering(n_clusters=100)  # 10 kümeye bölecek ?ekilde ayarlayabilirsiniz

# Modeli veriye fit edin
clusters = agg_clustering.fit_predict(X)

# Kümeleri al?n
filtered_messstellen_gw['cluster'] = clusters

# Scatter plot
plt.figure(figsize=(19, 10))
unique_clusters = np.unique(clusters)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_clusters)))  # Spectral renk paleti ile renkler al?n?r

# Her cluster için scatter plot
for i, cluster in enumerate(unique_clusters):
    cluster_points = X[clusters == cluster]
    plt.scatter(cluster_points['longitude'], cluster_points['latitude'],
                label=f'Cluster {cluster}', s=50, c=[colors[i]], marker='o')

# Etiketler
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Agglomerative Clustering of Locations')

# Legend ekleme
plt.legend(title="Clusters", loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()  # Legend'?n grafikle daha iyi yerle?mesini sa?lar

plt.show()



# Kaynaklar
messstellen_qu = pd.read_csv("datasets/messstellen_qu.csv", sep=';', encoding='windows-1252')

def convert_and_replace(df, x_col, y_col):
    """
    Verilen dataframe'deki belirli sütunlar? float tipine dönü?türür ve virgülleri nokta ile de?i?tirir.

    Args:
        df (pd.DataFrame): ??lem yap?lacak dataframe.
        x_col (str): Dönü?türülecek ilk sütunun ad?.
        y_col (str): Dönü?türülecek ikinci sütunun ad?.

    Returns:
        pd.DataFrame: Güncellenmi? dataframe.
    """
    df[x_col] = df[x_col].astype(str).str.replace(',', '.').astype(float)
    df[y_col] = df[y_col].astype(str).str.replace(',', '.').astype(float)
    return df

messstellen_qu = convert_and_replace(messstellen_qu, "xrkko09", "yhkko10")

messstellen_qu[['latitude', 'longitude']] = messstellen_qu.apply(lambda row: convert_coords(row['xrkko09'], row['yhkko10']), axis=1)

# görselle?tirme
import folium

# Örnek veri: Latitude ve Longitude
locations = messstellen_qu[['latitude', 'longitude']]

# Harita olu?tur
m = folium.Map(location=[locations['latitude'].mean(), locations['longitude'].mean()], zoom_start=8)

# Her bir nokta için Marker ekle
for _, row in locations.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Lat: {row['latitude']}, Lon: {row['longitude']}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

# Haritay? kaydet
m.save('austria_qu.html')

# Haritay? görüntüle (Jupyter Notebook kullan?yorsan?z bu sat?r? kullanabilirsiniz)
m



# Yerüstü Sular?
messstellen_owf = pd.read_csv("datasets/messstellen_owf.csv", sep=';', encoding='windows-1252')

def convert_and_replace(df, x_col, y_col):
    """
    Verilen dataframe'deki belirli sütunlar? float tipine dönü?türür ve virgülleri nokta ile de?i?tirir.

    Args:
        df (pd.DataFrame): ??lem yap?lacak dataframe.
        x_col (str): Dönü?türülecek ilk sütunun ad?.
        y_col (str): Dönü?türülecek ikinci sütunun ad?.

    Returns:
        pd.DataFrame: Güncellenmi? dataframe.
    """
    df[x_col] = df[x_col].astype(str).str.replace(',', '.').astype(float)
    df[y_col] = df[y_col].astype(str).str.replace(',', '.').astype(float)
    return df

messstellen_owf = convert_and_replace(messstellen_owf, "xrkko08", "yhkko09")

messstellen_owf[['latitude', 'longitude']] = messstellen_owf.apply(lambda row: convert_coords(row['xrkko08'], row['yhkko09']), axis=1)

# görselle?tirme
import folium

# Örnek veri: Latitude ve Longitude
locations = messstellen_owf[['latitude', 'longitude']]

# Harita olu?tur
m = folium.Map(location=[locations['latitude'].mean(), locations['longitude'].mean()], zoom_start=8)

# Her bir nokta için Marker ekle
for _, row in locations.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Lat: {row['latitude']}, Lon: {row['longitude']}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

# Haritay? kaydet
m.save('austria_owf.html')

# Haritay? görüntüle (Jupyter Notebook kullan?yorsan?z bu sat?r? kullanabilirsiniz)
m


