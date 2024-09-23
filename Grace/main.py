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

for key, df in monthly_gldas.items():
    # DataFrame'deki (lat, lon) sütunlar?na göre tuple olu?turuyoruz
    df['coord_tuple'] = list(zip(df['lat'], df['lon']))

    # DataFrame'i intersection_set'e göre filtreliyoruz
    filtered_df = df[df['coord_tuple'].apply(lambda x: x in intersection_set)]

    # Filtrelenmi? DataFrame'i yeni sözlü?e ekliyoruz
    filtered_dfs[key] = filtered_df.drop(columns=['coord_tuple'], inplace=True)

for key, df in filtered_dfs.items():
    df.reset_index(drop=True, inplace=True)

with open("Grace/pkl_files/gldas_dict_2010_2024.pkl", "wb") as f:
    pickle.dump(filtered_dfs, f)



# GRACE filtreleme, nan doldurma, sözlük yapma
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


# Merging Gldas and GRACE datasets
with open("Grace/pkl_files/gldas_dict_2010_2024.pkl", "rb") as f:
    gldas_dict_2010_2024 = pickle.load(f)

with open("Grace/pkl_files/grace_imputed_in_dict.pkl", "rb") as f:
    grace_dict = pickle.load(f)

with open("Grace/pkl_files/gldas_dict_2004_2009.pkl", "rb") as f:
    gldas_dict_2004_2009 = pickle.load(f)


# Merging Grace and Gladas
for key in gldas_dict_2010_2024.keys():
    # GLDAS ve GRACE verilerinin o aya ait DataFrame'lerini al
    gldas_df = gldas_dict_2010_2024[key]
    grace_df = grace_dict[key]

    if grace_df is not None:
        # GLDAS ve GRACE DataFrame'lerini lat ve lon sütunlar?na göre birle?tir (inner join)
        merged_df = gldas_df.merge(grace_df[['lat', 'lon', 'lwe_thickness']], on=['lat', 'lon'], how='inner')

        # Birle?tirilmi? DataFrame'i GLDAS dict'ine geri koy
        gldas_dict_2010_2024[key] = merged_df


# Veriyi 209 sat?rda bire dü?ürme fonksiyonu
def reduce_to_first_of_209(df):
    return df.iloc[::209, :]  # Her 209 sat?rdan birini (ilk sat?r?) al


def convert_cols(df, input_col):
    # Sütunun tipini anlamak için son k?sm? kontrol etme ('_tavg' veya '_acc')
    col_type = input_col.split("_")[-1]  # f-string hatas? düzeltildi

    # E?er sütun '_tavg' ile bitiyorsa, özel formül uygulay?n
    if col_type == "tavg":
        df[f"new_{input_col}"] = df[input_col] * 10800 * 8 * 30
        df.drop(input_col, axis=1, inplace=True)  # Eski sütunu kald?r
    # E?er sütun '_acc' ile bitiyorsa, farkl? bir formül uygulay?n
    elif col_type == "acc":
        df[f"new_{input_col}"] = df[input_col] * 8 * 30
        df.drop(input_col, axis=1, inplace=True)  # Eski sütunu kald?r



def process_data(dict):
    results_dict = {}

    for key, df in dict.items():
        # Her 209 sat?rdan birini al ve küçültülmü? DataFrame'i results_dict'e ekle
        results_dict[key] = reduce_to_first_of_209(df)
        results_dict[key].reset_index(drop=True, inplace=True)

    for month, df in results_dict.items():
        for col in df.columns:
            # Sadece '_tavg' veya '_acc' içeren sütunlar? dönü?tür
            if "_tavg" in col or "_acc" in col:
                convert_cols(df, col)

        # Yeni hesaplamalar yaparak sütunlar? ekleyin
        try:
            df['MSW'] = (df['new_Rainf_f_tavg'] + df['new_Qsb_acc']) - (df['new_Evap_tavg'] - df['new_ESoil_tavg'] + df['new_Qs_acc'])
        except KeyError as e:
            print(f"KeyError: {e}. Bu sütun eksik olabilir.")

        df.rename(columns={'SWE_inst': 'MSN'}, inplace=True)
        if 'lwe_thickness' in df.columns:
            df.rename(columns={'lwe_thickness': 'deltaTWS'}, inplace=True)

        # Toprak nemi ve s?cakl?k ortalamalar?n? hesaplay?n
        df['MSM'] = (df["SoilMoi0_10cm_inst"] + df["SoilMoi10_40cm_inst"] + df["SoilMoi40_100cm_inst"] +
                     df["SoilMoi100_200cm_inst"])

        df['SoilTMP0_avg'] = (df['SoilTMP0_10cm_inst'] + df['SoilTMP10_40cm_inst'] + df['SoilTMP40_100cm_inst'] +
                              df['SoilTMP100_200cm_inst'])

        # Kullan?lmayan sütunlar? kald?r?n
        cols_to_drop = ['SoilMoi0_10cm_inst', 'SoilMoi10_40cm_inst', 'SoilMoi40_100cm_inst', 'SoilMoi100_200cm_inst',
                        'SoilTMP0_10cm_inst', 'SoilTMP10_40cm_inst', 'SoilTMP40_100cm_inst', 'SoilTMP100_200cm_inst']
        df.drop(cols_to_drop, axis=1, inplace=True)

        # Güncellenmi? DataFrame'i sözlü?e geri yaz
        results_dict[month] = df

    return results_dict


results_dict_2010_2024 = process_data(gldas_dict_2010_2024)
results_dict_2004_2009 = process_data(gldas_dict_2004_2009)


with open('Grace/pkl_files/results_dict_2010_2024.pkl', 'wb') as file:
    pickle.dump(results_dict_2010_2024, file)

with open('Grace/pkl_files/results_dict_2004_2009.pkl', 'wb') as file:
    pickle.dump(results_dict_2004_2009, file)


# gldas2004-2009 daki 'MSW', 'MSM', 'MSN ortalamalar?n? hesapla:

# deliricem art?k yether