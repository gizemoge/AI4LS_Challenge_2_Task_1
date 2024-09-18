import pandas as pd
import xarray as xr
import pickle
from datetime import datetime, timedelta
"""
# Veri kümesini yükleme
ds = xr.open_dataset('Grace/datasets/(10)CSR_GRACE_GRACE-FO_RL0602_Mascons_all-corrections.nc')
lwe_thickness = ds["lwe_thickness"]

# Veriyi Pandas DataFrame'e dönü?türme
df_lwe = lwe_thickness.to_dataframe().reset_index()

# float16 veri türlerine dönü?türme (daha az bellek kullan?m? için)
df_lwe['time'] = df_lwe['time'].astype('float16')
df_lwe['lat'] = df_lwe['lat'].astype('float16')
df_lwe['lon'] = df_lwe['lon'].astype('float16')
df_lwe['lwe_thickness'] = df_lwe['lwe_thickness'].astype('float16')


# 'group' kolonunu ekleyerek her 20 sat?r? bir grup olarak i?aretleme
df_lwe['group'] = df_lwe.index // 20


# Gruplama i?lemi ve lon_range hesaplama
def compute_lon_range_and_agg(df_chunk):
    results = []
    for name, group in df_chunk.groupby('group'):
        lon_range = f"{group['lon'].min()} to {group['lon'].max()}"
        result = group.iloc[0:1].copy()
        result['lon_range'] = lon_range
        results.append(result)
    return pd.concat(results)

# Chunk size belirleme
chunk_size = 50000  # Özelle?tirilebilir; belle?inize uygun bir de?eri seçin

# Sonuçlar? saklamak için bir liste
results = []

# Veri kümesini chunk'lara bölme ve her chunk üzerinde i?lemi gerçekle?tirme
for start in range(0, len(df_lwe), chunk_size):
    end = min(start + chunk_size, len(df_lwe))
    df_chunk = df_lwe.iloc[start:end]
    df_chunk_grouped = compute_lon_range_and_agg(df_chunk)
    results.append(df_chunk_grouped)

# Tüm sonuçlar? birle?tirme
df = pd.concat(results).reset_index(drop=True)

# Sonuçlar? gösterme
df = df[['time', 'lat', 'lon_range', 'lwe_thickness']]
df.head(10)


# DataFrame'i pickle format?nda kaydetme
df.to_pickle('Grace/pkl_files/grace_float16_20.pkl')


# Pickle format?ndaki DataFrame'i yükleme
df = pd.read_pickle('Grace/pkl_files/grace_float16_20.pkl')
df.head(10)
"""

###################################################################################################################### 3
# LAND MASK
###################################################################################################################### 3
ds_land = xr.open_dataset('Grace/datasets/(3)CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc')
# land mask

ds_land
ds_land.variables
ds_land.data_vars
ds_land.coords

LO_val = ds_land["LO_val"]
LO_val.shape

df_land = ds_land['LO_val'].to_dataframe().reset_index()
df_land.head()

df_land["lat"].nunique()  # 720
df_land["lon"].nunique()  # 1440
df_land["LO_val"].nunique()  # 2

df_land["LO_val"].value_counts()
# LO_val
# 0.0    671474
# 1.0    365326
# Land=1 and Ocean=0

df_land_1 = df_land[df_land["LO_val"] == 1]
df_land_1["LO_val"].nunique()  # 1
df_land_1.head()

##################################################################################################################### 10
ds_lwe = xr.open_dataset('Grace/datasets/(10)CSR_GRACE_GRACE-FO_RL0602_Mascons_all-corrections.nc')

ds_lwe
ds_lwe.variables
ds_lwe.data_vars
ds_lwe.coords

ds_lwe.data_vars
# Data variables:
# time_bounds: Zaman dilimlerinin sinir bilgilerini içerir ve iki boyutu vardir: zaman ve sinir
# lwe_thickness: sivi su esdeger kalinligi verilerini içerir ve üç boyutu vardir: zaman, enlem, ve boylam

time_bounds_lwe = ds_lwe["time_bounds"]
lwe_thickness = ds_lwe["lwe_thickness"]

lwe_thickness.shape  # (232, 720, 1440) 232 zaman diliminde 720 enlem noktasi ve 1440 boylam noktasi içeriyor


##############
df_lwe = ds_lwe['lwe_thickness'].to_dataframe().reset_index()
df_lwe.head()
df_lwe.shape  # (240537600, 4)
df_lwe["time"].nunique()  # 232 yani 232 ay

df_lwe_time = ds_lwe['time_bounds'].to_dataframe().reset_index()
df_lwe_time.head()  # 0 v 1, bir ay?n aras?


# There are a total of 180 latitudes, ninety north, ninety south, and one is the equator. There is a total of 360 longitudes
df_lwe["lat"].nunique()  # 720
df_lwe["lon"].nunique()  # 1440
720/180  # 4
1440/360  # 4
# uydular dünyan?n etraf?nda 4 defa tam tur dönmüs olabilir mi?
# or her iki enlem aras? belli aral?klarla ölçüm yapm?? olabilir

df_lwe["lat"].unique()
# 720 unique enlem var ama
# her tam say?da 4 tane farkl? enlem var
# iki enlem aras? ?111.1 km

df_lwe["lon"].unique()
# 1440 unique boylam var

# 'df_lwe_time' ile 'df_lwe' merge (emin de?ilim çok)
merged_df = pd.merge(df_lwe.head(10), df_lwe_time.head(10), on='time')
merged_df

merged_df = pd.merge(df_lwe.tail(30), df_lwe_time.tail(30), on='time')
merged_df

########################################################################################################################
# kar??la?t?rma:
########################################################################################################################
import xarray as xr
import pandas as pd
ds_land = xr.open_dataset('Grace/datasets/(3)CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc')
df_land = ds_land['LO_val'].to_dataframe().reset_index()


ds = xr.open_dataset('Grace/datasets/(10)CSR_GRACE_GRACE-FO_RL0602_Mascons_all-corrections.nc')
lwe_thickness = ds["lwe_thickness"]

# Veriyi Pandas DataFrame'e dönü?türme
df_lwe = lwe_thickness.to_dataframe().reset_index()


df_land['LO_val'].shape[0] * 232
df_lwe.isnull().sum()

# Ayn? sütunu 232 kez tekrar et
df_land_expanded = pd.concat([df_land['LO_val']] * 232, ignore_index=True)

# E?er sadece LO_val sütunu geni?letilecekse ve df_lwe ile birle?tirilecekse
df_lwe_new = pd.concat([df_lwe, df_land_expanded], axis=1)
df_lwe_new = df_lwe_new.rename(columns={0: "land_mask"})
df_lwe_new.head()

# land_mask sütunu de?eri 1 olanlar? filtrele
df_filtered = df_lwe_new[df_lwe_new['land_mask'] == 1]
df_filtered.head()


df_filtered.shape

df_filtered.drop("land_mask", axis=1, inplace=True)
df_filtered.reset_index(drop=True, inplace=True)


########################################################################################################################
# tarihi düzenleme:
########################################################################################################################


# Ba?lang?ç tarihi
start_date = datetime.strptime('2002-01-01', '%Y-%m-%d')


def convert_time_to_date(time_value, start_date):
    return start_date + timedelta(days=time_value)

# 'time' de?erlerini tarihe dönü?tür
df_filtered['date'] = df_filtered['time'].apply(lambda x: convert_time_to_date(x, start_date))

# Tarihleri 'YYYY-MM-DD' format?nda yazd?r
df_filtered['date'] = df_filtered['date'].dt.strftime('%Y-%m-%d')
df_filtered = df_filtered[["date", "lat", "lon", "lwe_thickness"]]

df_filtered.to_pickle('Grace/pkl_files/df_grace_filtered.pkl')
########################################################################################################################

df_filtered = pd.read_pickle("Grace/pkl_files/df_grace_filtered.pkl")

df_filtered.head()





