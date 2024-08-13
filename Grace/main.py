import pandas as pd
import xarray as xr
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

###################################################################################################################### 1
ds = xr.open_dataset('Grace/datasets/(1)CSR_GRACE-FO_RL0602_Mascons_MasconC30-component.nc')
# ds = dataset

ds
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
df_lwe.head()
df_lwe.shape  # (240537600, 4) OHA

df_lwe["time"].nunique()  # 232
# Mart 2002'den bu yana toplam 269 ay geçti?

# dünyada 90 enlem 180 boylam var
df_lwe["lat"].nunique()  # 720
df_lwe["lon"].nunique()  # 1440
720/90
1440/180
# uydular dünyan?n etraf?nda 8 defa tam tur dönmü? olabilir mi?


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


###################################################################################################################### 2
ds_2 = xr.open_dataset('Grace/datasets/(2)CSR_GRACE-FO_RL0602_Mascons_SLR-C30-component.nc')

ds_2
ds_2.variables
ds_2.data_vars
ds_2.coords

ds_2.attrs

###################################################################################################################### 3
ds_3 = xr.open_dataset('Grace/datasets/(3)CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc')
# land mask

ds_3
ds_3.variables
ds_3.data_vars
ds_3.coords
ds_3.attrs

LO_val = ds_3["LO_val"]
LO_val.shape

df_3 = ds_3['LO_val'].to_dataframe().reset_index()
df_3.head()

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

###################################################################################################################### 5
ds_5 = xr.open_dataset('Grace/datasets/(5)CSR_GRACE_GRACE-FO_RL0602_Mascons_GAD-component.nc')

ds_5
ds_5.variables
ds_5.data_vars
ds_5.coords
ds_5.attrs

###################################################################################################################### 6
ds_6 = xr.open_dataset('Grace/datasets/(6)CSR_GRACE_GRACE-FO_RL0602_Mascons_GIA-component.nc')

ds_6
ds_6.variables
ds_6.data_vars
ds_6.coords
ds_6.attrs

###################################################################################################################### 7
ds_7 = xr.open_dataset('Grace/datasets/(7)CSR_GRACE_GRACE-FO_RL0602_Mascons_GSU-component.nc')

ds_7
ds_7.variables
ds_7.data_vars
ds_7.coords
ds_7.attrs


###################################################################################################################### 8
ds_8 = xr.open_dataset('Grace/datasets/(8)CSR_GRACE_GRACE-FO_RL0602_Mascons_MasconC20-component.nc')

ds_8
ds_8.variables
ds_8.data_vars
ds_8.coords
ds_8.attrs


###################################################################################################################### 9
ds_9 = xr.open_dataset('Grace/datasets/(9)CSR_GRACE_GRACE-FO_RL0602_Mascons_SLR-C20-component.nc')

ds_9
ds_9.variables
ds_9.data_vars
ds_9.coords
ds_9.attrs


##################################################################################################################### 10
ds_10 = xr.open_dataset('Grace/datasets/(10)CSR_GRACE_GRACE-FO_RL0602_Mascons_all-corrections.nc')

ds_10
ds_10.variables
ds_10.data_vars
ds_10.coords
ds_10.attrs


##################################################################################################################### 11
ds_11 = xr.open_dataset('Grace/datasets/(11)CSR_GRACE_GRACE-FO_RL0602_Mascons_degree1-component.nc')

ds_11
ds_11.variables
ds_11.data_vars
ds_11.coords
ds_11.attrs

