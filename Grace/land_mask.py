import pandas as pd
import xarray as xr
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

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

ds.data_vars
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