import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pyproj import Transformer
from scipy.spatial import cKDTree

pd.set_option('display.max_columns', None)

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

#filtered_messstellen_nvl.to_csv('datasets/filtered_messstellen_nvl.csv', index=False, encoding='windows-1252')
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


# yukar?daki dönü?türme ve float yapma i?lemlerini nas?l tek fonk ile hallederiz :

def transform_coordinates(df):
    # x ve y ile ba?layan sütun isimlerini bulma
    x_cols = [col for col in df.columns if col.startswith('x')]
    y_cols = [col for col in df.columns if col.startswith('y')]

    # Sütunlarda string dönü?ümü ve nokta ile de?i?tirip float yapma
    for col in x_cols + y_cols:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    # Transformer tan?mlama
    transformer = Transformer.from_crs("EPSG:31287", "EPSG:4326", always_xy=True)

    # Dönü?türülmü? koordinatlar? ekleme
    for x_col, y_col in zip(x_cols, y_cols):
        coords = df.apply(lambda row: transformer.transform(row[x_col], row[y_col]), axis=1)
        df['longitude'] = coords.apply(lambda coord: coord[0])
        df['latitude'] = coords.apply(lambda coord: coord[1])

    return df

# burada filtrelenmemei? hallerini de koordinatlar? ekleyece?im:
filtered_messstellen_gw = pd.read_csv("datasets/filtered_messstellen_gw.csv", sep=';', encoding='windows-1252')
messstellen_gw = pd.read_csv("datasets/messstellen_gw.csv", sep=';', encoding='windows-1252')
messstellen_nlv = pd.read_csv("datasets/messstellen_nlv.csv", sep=';', encoding='windows-1252')
messstellen_owf = pd.read_csv("datasets/messstellen_owf.csv", sep=';', encoding='windows-1252')
messstellen_qu = pd.read_csv("datasets/messstellen_qu.csv", sep=';', encoding='windows-1252')

transformed_filtered_messstellen_gw = transform_coordinates(filtered_messstellen_gw)
transformed_messstellen_gw = transform_coordinates(messstellen_gw)
transformed_messstellen_nlv = transform_coordinates(messstellen_nlv)
transformed_messstellen_owf = transform_coordinates(messstellen_owf)
transformed_messstellen_qu = transform_coordinates(messstellen_qu)

# kay?tl? çok az csv varm?? gibi bi de bunu kaydettim:
transformed_filtered_messstellen_gw.to_csv('datasets/transformed_filtered_messstellen_gw.csv', index=False)
transformed_messstellen_gw.to_csv('datasets/transformed_messstellen_gw.csv', index=False)
transformed_messstellen_nlv.to_csv('datasets/transformed_messstellen_nlv.csv', index=False)
transformed_messstellen_owf.to_csv('datasets/transformed_messstellen_owf.csv', index=False)
transformed_messstellen_qu.to_csv('datasets/transformed_messstellen_qu.csv', index=False)

# en yak?n koordinatlar? bulmak için cKDTree diye bi?ey varm?? onu deniyorum:




# Groundwater ve rain veri setlerindeki koordinatlar? numpy array'e çevirme
groundwater_coords = transformed_messstellen_gw[['longitude', 'latitude']].values
nlv_coords = transformed_messstellen_nlv[['longitude', 'latitude']].values

# KDTree yap?s?n? olu?turma
tree = cKDTree(nlv_coords)

# Her bir groundwater noktas? için en yak?n rain noktas?n? bulma
distances, indices = tree.query(groundwater_coords, k=1)

# En yak?n rain noktalar?n? groundwater veri setine ekleme
transformed_messstellen_gw['nearest_rain_index'] = indices
transformed_messstellen_gw['nearest_rain_distance'] = distances

# En yak?n rain verilerini groundwater veri setine ekleme
nearest_rain_data = transformed_messstellen_nlv.iloc[indices].reset_index()
transformed_messstellen_gw = transformed_messstellen_gw.reset_index().join(nearest_rain_data, rsuffix='_rain')
transformed_messstellen_gw.head()
# Gereksiz sütunlar? kald?rma ve yeniden adland?rma

# BURADA HANG? SÜTUNLARI KALDIRMALI HANG?S? KALMALI YOKSA SADECE E?LE?T?R?LEN HZBNR LER? M? ALMALI B?LEMED?M

# transformed_messstellen_gw.drop(columns=['nearest_rain_index', 'nearest_rain_distance', 'Date_rain'], inplace=True)
# transformed_messstellen_gw.rename(columns={'rain': 'nearest_rain'}, inplace=True)


# bi?ey deniyorum bu benim daha önceki e?le?tirmeme benzedi mi?

new_hzbnr = transformed_messstellen_gw[["hzbnr01", "hzbnr01_rain"]]

hzbnr = pd.read_csv("datasets/hzbnr.csv", sep=';')

hzbnr.equals(new_hzbnr)

# tabi bu daha büyük oldu?u için ayn? ç?kmad? ama bunu dü?ünücem
