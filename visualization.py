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
# İstenen tüm lokasyonları haritada görelim:

# Veriyi yükle
filtered_messstellen_gw = pd.read_csv("datasets/filtered_messstellen_gw.csv", sep=',', encoding='windows-1252')

# Koordinat sütunlarını doğru formata dönüştür
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

