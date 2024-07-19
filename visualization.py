# imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.dates as mdates
import geopandas as gpd
from shapely.geometry import Point
import math
from pyproj import Transformer
pd.set_option('display.max_columns', None)

############################################################################################################
# İstenen tüm lokasyonları haritada görelim:

# Veriyi yükle
filtered_messstellen_gw = pd.read_csv("datasets/filtered_messstellen_gw.csv", sep=',', encoding='windows-1252')

# Koordinat sütunlarını doğru formata dönüştür
coordinates_of_locations = filtered_messstellen_gw[['hzbnr01', 'xrkko09', 'yhkko10']]
coordinates_of_locations['xrkko09'] = coordinates_of_locations['xrkko09'].str.replace(',', '.').astype(float)
coordinates_of_locations['yhkko10'] = coordinates_of_locations['yhkko10'].str.replace(',', '.').astype(float)

# Bessel 1841 ve WGS 84 (GPS) için dönüşüm tanımlayıcıları
transformer = Transformer.from_crs('epsg:4312', 'epsg:4326')

# Koordinatları dönüştür ve yeni sütunları oluştur
def transform_coordinates(row):
    lon, lat = transformer.transform(row['xrkko09'], row['yhkko10'])
    return pd.Series({'longitude': lon, 'latitude': lat})

coordinates_of_locations[['longitude', 'latitude']] = coordinates_of_locations.apply(transform_coordinates, axis=1)




coordinates_of_locations.head()









# Koordinatları GeoDataFrame'e dönüştürme
geometry = [Point(xy) for xy in zip(coordinates_of_locations['xrkko09_dms'], coordinates_of_locations['yhkko10_dms'])]
gdf = gpd.GeoDataFrame(coordinates_of_locations, geometry=geometry)

# Koordinat referans sistemi (CRS) ayarı (örneğin EPSG:31287 - Avusturya projeksiyon sistemi)
gdf.crs = 'EPSG:31287'

# GeoDataFrame'i gösterme
print(gdf)

# Haritayı oluştur
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color='red', markersize=50)  # Noktaları kırmızı renkte ve büyük boyutta göster

# Haritayı göster
plt.show()


############################################################################################################
# Örnek bir lokasyonun yer altı su seviyesi
df = pd.read_csv("datasets/processed/300111.csv", sep=';', encoding='windows-1252')
df.shape
df.head()

# Tarih sütununu datetime formatına çevirme
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Seaborn ile grafik çizimi
plt.figure(figsize=(18, 8))
sns.lineplot(x='Date', y='Value', data=df, color='b')

# x ekseninde her 5 yılda bir yıl gösterme
plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.title('Zaman Serisi Grafiği')
plt.xlabel('Tarih')
plt.ylabel('Değer')
plt.grid(True)

# Tarih etiketlerinin daha iyi gözükmesi için otomatik düzenleme
plt.gcf().autofmt_xdate()

# Grafiği göster
plt.show()

#####################################################################################################################