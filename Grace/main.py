import modin.pandas as pd
import xarray as xr
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Grace verisini açt?k, karalar? filtreledik
df_land = xr.open_dataset('Grace/datasets/(3)CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc')
df_land = df_land['LO_val'].to_dataframe().reset_index()


df_lwe = xr.open_dataset('Grace/datasets/(10)CSR_GRACE_GRACE-FO_RL0602_Mascons_all-corrections.nc')
df_lwe = df_lwe["lwe_thickness"].to_dataframe().reset_index()

# Ayn? sütunu 232 kez tekrar et
df_land_expanded = pd.concat([df_land['LO_val']] * 232, ignore_index=True)

# E?er sadece LO_val sütunu geni?letilecekse ve df_lwe ile birle?tirilecekse
df = pd.concat([df_lwe, df_land_expanded], axis=1)

# land_mask sütunu de?eri 1 olanlar? filtrele
df = df[df['LO_val'] == 1]

df.drop("LO_val", axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)


# time k?sm?n? tarih fromat?na geçirdik
start_date = datetime.strptime('2002-01-01', '%Y-%m-%d')


def convert_time_to_date(time_value, start_date):
    return start_date + timedelta(days=time_value)

# 'time' de?erlerini tarihe dönü?tür
df['time'] = df['time'].apply(lambda x: convert_time_to_date(x, start_date))


df['lon'] = df['lon'].apply(lambda x: x - 360 if x > 180 else x)

# time sütununu datetime format?na dönü?tür
df['time'] = pd.to_datetime(df['time'])

# 2010 y?l?ndan önceki sat?rlar? filtrele ve kald?r
df = df[df['time'] >= '2010-01-01']

df.reset_index(drop=True, inplace=True)

# time sütunundan sadece tarih k?sm?n? almak
df['time'] = df['time'].dt.date


# koordinatlar kontrol ediliyor her ayda ayn? düzendeler mi diye
# ?lk ay? referans almak için ilk lat-lon çiftlerini çek
first_month_coords = set(zip(df[df['time'].dt.to_period('M') == df['time'].dt.to_period('M').iloc[0]]['lat'],
                             df[df['time'].dt.to_period('M') == df['time'].dt.to_period('M').iloc[0]]['lon']))

# Her ay? teker teker kontrol etmek
all_same = True  # Ba?lang?çta ayn? oldu?unu varsay?yoruz
for year_month, group in df.groupby(df['time'].dt.to_period('M')):
    coords = set(zip(group['lat'], group['lon']))
    if coords != first_month_coords:
        all_same = False
        break

# Sonuç olarak evet veya hay?r yazd?rma
if all_same:
    print("Evet")
else:
    print("Hay?r")


# gladas verisini açma
with open('Grace/pkl_files/gldas_dict_2010_2024.pkl', 'rb') as file:
    monthly_gldas = pickle.load(file)

# Gldas'taki tüm aylara ait koordinatlar ayn? m? Evet
# Her bir DataFrame içindeki lat-lon çiftlerini toplay?p bir set'e ekleme
coordinates_per_df = [set(zip(df['lat'], df['lon'])) for df in monthly_gldas.values()]

# ?lk seti referans alarak di?er setler ile kar??la?t?rma
all_same = all(coords == coordinates_per_df[0] for coords in coordinates_per_df)

# Sonuç olarak evet veya hay?r yazd?rma
if all_same:
    print("Evet")
else:
    print("Hay?r")


# Intersection of latitude and longitude couples that come from Gldas and GRACE datasets.
intersection_set = first_month_coords.intersection(coordinates_per_df[0])


# Editing the coordinates in GLDAS according to the intersection set.
# Filtrelenmi? DataFrame'leri saklayacak bir sözlük olu?turuyoruz
filtered_dfs = {}

# Her bir DataFrame için filtreleme i?lemi
for key, df in monthly_gldas.items():
    # DataFrame'deki (lat, lon) sütunlar?na göre tuple olu?turuyoruz
    df['coord_tuple'] = list(zip(df['lat'], df['lon']))

    # DataFrame'i intersection_set'e göre filtreliyoruz
    filtered_df = df[df['coord_tuple'].apply(lambda x: x in intersection_set)]

    # Filtrelenmi? DataFrame'i yeni sözlü?e ekliyoruz
    filtered_dfs[key] = filtered_df

    # 'coord_tuple' sütununu kald?r?yoruz (filtre i?lemi bitti?i için gerek kalmad?)
    filtered_dfs[key].drop(columns=['coord_tuple'], inplace=True)

monthly_gldas_edited = filtered_dfs.copy()

for key, df in monthly_gldas_edited.items():
    df.reset_index(drop=True, inplace=True)


# Editing the coordinates in GRACE according to the intersection set.
df = df[df[['lat', 'lon']].apply(tuple, axis=1).isin(intersection_set)]

df.reset_index(drop=True, inplace=True)

# Imputing NaN values
df["time"] = df["time"].apply(lambda x: x.replace(day=1))



# Time sütununu datetime format?na çevir
df['time'] = pd.to_datetime(df['time'])

# 2010-01-01 ay?ndaki lat-lon kombinasyonlar?n? al
reference_lat_lon = df[df['time'] == '2010-01-01'][['lat', 'lon']].drop_duplicates()

# Tüm mevcut aylar? tespit et
existing_months = df['time'].drop_duplicates()

# 2010 ve 2024 y?llar?ndaki tüm aylar? belirle
all_months = pd.date_range(start='2010-01-01', end='2024-12-01', freq='MS')

# Eksik aylar? tespit et
missing_months = all_months.difference(existing_months)

# Eksik aylar için lat-lon kombinasyonlar?n? kullanarak yeni sat?rlar ekle
missing_data = pd.concat(
    [pd.DataFrame({
        'time': [month] * len(reference_lat_lon),  # Eksik olan her ay için lat-lon kombinasyonlar? ekleniyor
        'lat': reference_lat_lon['lat'].values,
        'lon': reference_lat_lon['lon'].values,
        'lwe_thickness': np.nan})  # lwe_thickness sütunu NaN olarak ekleniyor
     for month in missing_months]
)

# Eksik aylar? orijinal verilerle birle?tirip s?ralama yap?yoruz
df_filled_corrected = pd.concat([df, missing_data]).drop_duplicates(subset=['time', 'lat', 'lon']).sort_values(by=['time', 'lat', 'lon']).reset_index(drop=True)



# GRACE dataframe to dictionary
# 'year-month' format?nda anahtar olu?tur
df_filled_corrected['key'] = df_filled_corrected['time'].dt.strftime('%Y%m')

# Sözlük olu?tur
result_dict = {key: group.drop(columns='key') for key, group in df_filled_corrected.groupby('key')}


for key, value in result_dict.items():
    value.reset_index(inplace=True, drop=True)



# Imputing NaN Values
# NaN de?erleri doldurmak için
for month_key, month_df in result_dict.items():
    # Anahtar?n ay k?sm?n? al
    current_month = month_key[-2:]  # Ay k?sm?n? al
    measurement_index = month_df.index  # Ölçüm noktas? indeksleri

    # Her ölçüm noktas? için
    for i in measurement_index:
        if pd.isna(month_df.at[i, 'lwe_thickness']):  # NaN kontrolü
            # Di?er y?llardaki o ay verilerini toplamak için liste olu?tur
            other_year_values = []
            for year in range(2010, 2025):  # 2010'dan 2024'e kadar
                year_key = f"{year}{current_month}"
                if year_key in result_dict:
                    other_year_df = result_dict[year_key]
                    if i < len(other_year_df):  # Ölçüm noktas? indeksinin geçerli olup olmad???n? kontrol et
                        value = other_year_df.at[i, 'lwe_thickness']
                        if pd.notna(value):
                            other_year_values.append(value)

            # E?er de?erler varsa, ortalamay? hesapla ve NaN olan yere yaz
            if other_year_values:
                average_value = np.mean(other_year_values)
                month_df.at[i, 'lwe_thickness'] = average_value


with open('Grace/pkl_files/grace_imputed_in_dict.pkl', 'wb') as f:
    pickle.dump(result_dict, f)


# todo grace 2024 nisan sonras?n? droplaman?z gerek sözlükte