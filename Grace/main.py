import pandas as pd
import xarray as xr
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
import numpy as np

###################################################################################################################### 1
ds = xr.open_dataset('Grace/datasets/(10)CSR_GRACE_GRACE-FO_RL0602_Mascons_all-corrections.nc')
# ds = dataset

import pprint
pprint.pprint(ds.attrs)


ds.info()
# ds metadatay? bulunduruyor
# 4 section var: Dimensions, Coordinates, Data Variables ve Attributes


ds.coords
# Verilerin düzenlenmesi için kullanilan boyutlarin isimleri ve bu boyutlara karsilik gelen degerler.
# Örnegin, zaman, enlem, boylam gibi koordinatlar olabilir.
# Baslarinda yildiz var demek, birden çok data variable'?nda kullnailiyor demek.
# Coordinates:
#   * time     (time) float32 107.0 129.5 227.5 ... 8.08e+03 8.11e+03 8.141e+03 (time da koordinat olarak say?l?yor)
#   * lon      (lon) float32 0.125 0.375 0.625 0.875 ... 359.1 359.4 359.6 359.9
#   * lat      (lat) float32 -89.88 -89.62 -89.38 -89.12 ... 89.38 89.62 89.88

ds.data_vars
# Data variables:
#     time_bounds    (time, timebound) float32 ...
#     lwe_thickness  (time, lat, lon) float32 ...
# time_bounds: Zaman dilimlerinin sinir bilgilerini içerir ve iki boyutu vardir: zaman ve sinir.
# lwe_thickness: sivi su esdeger kalinligi verilerini içerir ve üç boyutu vardir: zaman, enlem, ve boylam.


# Her variable'a bu ?ekilde ula??labiliyor:
time_bounds = ds["time_bounds"]
lwe_thickness = ds["lwe_thickness"]

lwe_thickness.shape  # (232, 720, 1440) 232 zaman diliminde 720 enlem noktasi ve 1440 boylam noktasi içeriyor
lwe_thickness.mean()


# ?lk time index'ini seçmek:
lwe_thickness[0,:]

# Plotting
lwe_thickness.isel(time=0).plot(size=6)

lwe_thickness.sel(time=slice('107.0','227.5'), lon=slice(20,160), lat=slice(-80,25)).plot(size=6)

lwe_thickness.sel(time=slice('107.0','227.5')).sel(lat=-27.47, lon=153.03, method='nearest').plot(size=6)




##############
df_lwe = ds['lwe_thickness'].to_dataframe().reset_index()
df_lwe.head(20)
df_lwe.shape  # (240537600, 4)

df_lwe["time"].nunique()  # 232
# Mart 2002'den bu yana toplam 269 ay geçti?

# dünyada 90 enlem 180 boylam var
df_lwe["lat"].nunique()  # 720
df_lwe["lon"].nunique()  # 1440
df_lwe["lwe_thickness"].nunique()

720/180
1440/360
# uydular dünyan?n etraf?nda 8 defa tam tur dönmü? olabilir mi?

#########################################################
# bu noktada bir gruplama yap?p veriyi küçültmeyi deniyoruz
############################################################
#ba?ka bir py dosyas?nda deveam etcem










################################################################




df_tb = ds['time_bounds'].to_dataframe().reset_index()
df_tb.head()
# timebound'daki 0 zaman diliminin ba?lag?c?, 1 ise biti?i


# 'df_tb' ile 'df_lwe' tablolar?n? birle?tirme
merged_df = pd.merge(df_lwe.head(30), df_tb.head(30), on='time')
# ?lk birkaç sat?r? görüntüleme
print(merged_df)

# 'df_tb' ile 'df_lwe' tablolar?n? birle?tirme
merged_df = pd.merge(df_lwe.tail(30), df_tb.tail(30), on='time')
# ?lk birkaç sat?r? görüntüleme
print(merged_df)









##################################################
# Ba?lang?ç tarihi
start_date = pd.Timestamp("2002-01-01")

# time verisi
time_values = [107.0, 129.5, 227.5]  # Örnek zaman verileri

# Zaman? tarihe çevir
date_times = pd.to_timedelta(time_values, unit='D') + start_date
print(date_times)

##############################
time_bounds_values = [[94.0, 120.0], [122.0, 137.0], [212.0, 227.5]]  # Örnek time_bounds verisi

# time_bounds'? tarihe çevir
date_bounds = [(pd.to_timedelta(bounds[0], unit='D') + start_date,
                pd.to_timedelta(bounds[1], unit='D') + start_date)
                for bounds in time_bounds_values]
print(date_bounds)

###################################################################################################################### 3
ds_3 = xr.open_dataset('Grace/datasets/(3)CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc')
# land mask

ds_3.info()
ds_3.variables
ds_3.data_vars
ds_3.coords
ds_3.attrs

LO_val = ds_3["LO_val"]
LO_val.shape

df_3 = ds_3['LO_val'].to_dataframe().reset_index()
df_3.head(30)
df_3.shape
df_3["lat"].nunique()  # 720
df_3["lon"].nunique()  # 1440
df_3["LO_val"].nunique()  # 2

df_3["LO_val"].value_counts()
# LO_val
# 0.0    671474
# 1.0    365326
# Land=1 and Ocean=0

###################################################################################################################### 4
ds_4 = xr.open_dataset('Grace/datasets/(4)CSR_GRACE_GRACE-FO_RL06_Mascons_v02_OceanMask.nc')
# ocean mask

ds_4
ds_4.variables
ds_4.data_vars
ds_4.coords
ds_4.attrs

########################################################################################################################
# 'lwe_thickness' de?i?kenini DataFrame'e dönü?türme
df_lwe = ds['lwe_thickness'].to_dataframe().reset_index()

# DataFrame'in ilk birkaç sat?r?n? görüntüleme
print(df_lwe.head(30))


# LWE: Liquid water equivalent thickness


# 'time_bounds' de?i?kenini DataFrame'e dönü?türme
df_tb = ds['time_bounds'].to_dataframe().reset_index()

# DataFrame'in ilk birkaç sat?r?n? görüntüleme
print(df_tb.head(20))