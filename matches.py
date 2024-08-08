# bu dosyada nlv_rain nlw_snow owf ve qu ölçüm noktalar?n?n bizim istedi?imiz noktalara en yak?n 5 kom?usunu tespit ettik


import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


pd.set_option('display.max_columns', None)

filtered_messstellen_gw = pd.read_csv("datasets/transformed_filtered_messstellen_gw.csv")
filtered_messstellen_gw.head()

gw_coordinates = filtered_messstellen_gw[["hzbnr01", "latitude", "longitude"]]
gw_coordinates.head()
gw_coordinates.shape # (487,3)

#burada bunlar? kullanaca??m çünkü en yak?n tek bir nokta ile ili?kilendirmi?tim daha önceki denemelerimde ama o zaman e?le?tirme k?s?tl? olmu?tu.
transformed_messstellen_nlv_rain = pd.read_csv("datasets/transformed_nlv_rain.csv")
transformed_messstellen_nlv_rain.head()
nlv_rain_coordinates = transformed_messstellen_nlv_rain[["hzbnr01", "latitude", "longitude"]]
nlv_rain_coordinates.head()
nlv_rain_coordinates.shape # (863,3)

transformed_messstellen_nlv_snow = pd.read_csv("datasets/transformed_nlv_snow.csv")
transformed_messstellen_nlv_snow.head()
nlv_snow_coordinates = transformed_messstellen_nlv_snow[["hzbnr01", "latitude", "longitude"]]
nlv_snow_coordinates.head()
nlv_snow_coordinates.shape # (631,3)


transformed_messstellen_owf = pd.read_csv("datasets/transformed_messstellen_owf.csv")
transformed_messstellen_owf.head()
owf_coordinates = transformed_messstellen_owf[["hzbnr01", "latitude", "longitude"]]
owf_coordinates.head()
owf_coordinates.shape # (792,3)

transformed_messstellen_qu = pd.read_csv("datasets/transformed_messstellen_qu.csv")
transformed_messstellen_qu.head()
qu_coordinates = transformed_messstellen_qu[["hzbnr01", "latitude", "longitude"]]
qu_coordinates.head()
qu_coordinates.shape # (93,3)

# Parametreler
num_neighbors = 5  # En yak?n kaç kom?unun hesaba kat?laca??

# Yer alt? suyu seviyeleri noktalar?n? haz?rlama
groundwater_coords = np.array(list(zip(gw_coordinates['latitude'], gw_coordinates['longitude'])))

# Ya?mur ve kar verileri noktalar?n? haz?rlama
nlv_coords = np.array(list(zip(nlv_rain_coordinates['latitude'], nlv_rain_coordinates['longitude'])))
nlv_rain_tree = cKDTree(nlv_coords)
nlv_coords = np.array(list(zip(nlv_snow_coordinates['latitude'], nlv_snow_coordinates['longitude'])))
nlv_snow_tree = cKDTree(nlv_coords)

# ve di?erleri:
owf_coords = np.array(list(zip(owf_coordinates['latitude'], owf_coordinates['longitude'])))
owf_tree = cKDTree(owf_coords)
qu_coords = np.array(list(zip(qu_coordinates['latitude'], qu_coordinates['longitude'])))
qu_tree = cKDTree(qu_coords)




# En yak?n kom?ular? bulma
distances_nlv_rain, indices_nlv_rain = nlv_rain_tree.query(groundwater_coords, k=num_neighbors)
distances_nlv_snow, indices_nlv_snow = nlv_snow_tree.query(groundwater_coords, k=num_neighbors)
distances_owf, indices_owf = owf_tree.query(groundwater_coords, k=num_neighbors)
distances_qu, indices_qu = qu_tree.query(groundwater_coords, k=num_neighbors)

# A??rl?kl? ortalama hesaplama (ters mesafe)
weights_nlv_rain = 1 / distances_nlv_rain
weights_nlv_rain /= weights_nlv_rain.sum(axis=1, keepdims=True)

weights_nlv_snow = 1 / distances_nlv_snow
weights_nlv_snow /= weights_nlv_snow.sum(axis=1, keepdims=True)

weights_owf = 1 / distances_owf
weights_owf /= weights_owf.sum(axis=1, keepdims=True)

weights_qu = 1 / distances_qu
weights_qu /= weights_qu.sum(axis=1, keepdims=True)

# Sonuçlar? bir DataFrame'e ekleme
columns = (['hzbnr01'] + [f'nearest_nlv_rain_{i+1}' for i in range(num_neighbors)] +
           [f'nlv_rain_weight_{i+1}' for i in range(num_neighbors)] +
           [f'nearest_nlv_snow_{i+1}' for i in range(num_neighbors)] +
           [f'nlv_snow_weight_{i+1}' for i in range(num_neighbors)] +
           [f'nearest_owf_{i+1}' for i in range(num_neighbors)] +
           [f'owf_weight_{i+1}' for i in range(num_neighbors)] +
           [f'nearest_qu_{i+1}' for i in range(num_neighbors)] +
           [f'qu_weight_{i+1}' for i in range(num_neighbors)])
result_df = pd.DataFrame(columns=columns)

result_df['hzbnr01'] = gw_coordinates['hzbnr01']
for i in range(num_neighbors):
    result_df[f'nearest_nlv_rain_{i+1}'] = nlv_rain_coordinates['hzbnr01'].iloc[indices_nlv_rain[:, i]].values
    result_df[f'nlv_rain_weight_{i+1}'] = weights_nlv_rain[:, i]
    result_df[f'nearest_nlv_snow_{i+1}'] = nlv_snow_coordinates['hzbnr01'].iloc[indices_nlv_snow[:, i]].values
    result_df[f'nlv_snow_weight_{i+1}'] = weights_nlv_snow[:, i]
    result_df[f'nearest_owf_{i+1}'] = owf_coordinates['hzbnr01'].iloc[indices_owf[:, i]].values
    result_df[f'owf_weight_{i+1}'] = weights_owf[:, i]
    result_df[f'nearest_qu_{i+1}'] = qu_coordinates['hzbnr01'].iloc[indices_qu[:, i]].values
    result_df[f'qu_weight_{i+1}'] = weights_qu[:, i]

print(result_df)

result_df.to_csv('datasets/matches.csv', sep=';', index=False)
