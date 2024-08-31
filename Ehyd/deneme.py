# IMPORTS
import os
import ast
import warnings
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)
warnings.simplefilter('ignore', category=ConvergenceWarning)
from datetime import datetime
from scipy.spatial import distance
from collections import Counter
import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Pickle dosyalar?n? toplu açma i?lemi:
pkl_files = [f for f in os.listdir() if f.endswith('.pkl')]

for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as file:
        var_name = pkl_file[:-4]
        globals()[var_name] = pickle.load(file)

data = pd.read_csv("data.csv")



#######################################################################
# globals kulland???m?z için k?rm?z?lar
# sözlüklerin içindeki serileri dataframe yap?yorum.
dict_list = [filled_conductivity_dict, filled_data_gw_temp_dict, filled_groundwater_dict, filled_rain_dict,
             filled_sediment_dict_monthly, filled_snow_dict_monthly, filled_source_flow_rate_dict,
             filled_source_temp_dict, filled_surface_water_flow_rate_dict_monthly,
             filled_surface_water_level_dict_monthly, filled_surface_water_temp_dict]


# Dict list içindeki her bir sözlü?ün value'lar?n? DataFrame yapacak fonksiyon
def convert_series_to_dataframe(d):
    for key in d:
        d[key] = d[key].to_frame(name=key)
    return d

for i in range(len(dict_list)):
    dict_list[i] = convert_series_to_dataframe(dict_list[i])


# Lag ve rolling mean hesaplamalar?n? gerçekle?tirecek fonksiyon
def add_lag_and_rolling_mean(df, window=6):
    # ?lk sütunun ad?n? al
    column_name = df.columns[0]
    # 1 lag'li versiyonu ekle
    df[f'lag_1'] = df[column_name].shift(1)
    # Lag'li ve rolling mean sütunlar?n? ekle
    for i in range(1, 2):  # Burada 1 lag'li oldu?u için range(1, 2) kullan?yoruz.
        df[f'rolling_mean_{window}_lag_{i}'] = df[f'lag_1'].shift(i).rolling(window=window).mean()
    return df


for d in dict_list:
    for key, df in d.items():
        d[key] = add_lag_and_rolling_mean(df)

# zero padding
def zero_padding(df, start_date='1960-01-01'):
    # E?er indeks zaten PeriodIndex de?ilse, to_period('M') yap
    if not isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_period('M')

    # Belirlenen ba?lang?ç tarihi
    start_date = pd.to_datetime(start_date).to_period('M')

    # Tarih aral???n? geni?let
    all_dates = pd.period_range(start=start_date, end=df.index.max(), freq='M')

    # Yeni tarih aral??? için bo? bir veri çerçevesi olu?tur
    new_df = pd.DataFrame(index=all_dates)

    # Eski veri çerçevesini yeni veri çerçevesine birle?tir
    new_df = new_df.join(df, how='left').fillna(0)

    # Periyotlar? datetime'e dönü?tür
    new_df.index = new_df.index.to_timestamp()

    return new_df

# Her bir sözlükteki veri çerçevelerini güncelleme
for dictionary in dict_list:
    for key in dictionary:
        dictionary[key] = zero_padding(dictionary[key])


###############
# 11 sözlükteki tüm dataframe'lerin tüm kolonlar?n? float32'ye çevirme
def convert_to_float32(df):
    return df.astype('float32')

# Her bir sözlükteki veri çerçevelerini veri tipini float32'ye çevirme
for dictionary in dict_list:
    for key in dictionary:
        # Veri tipini float32'ye çevir
        dictionary[key] = convert_to_float32(dictionary[key])


################333

##############3
import ast
import pandas as pd

index_values = data['hzbnr01']

# hzbnr01 de?erlerini indeks olarak kullanan bo? bir DataFrame olu?tur
new_df = pd.DataFrame(index=index_values)
# Genel amaçl? veri çekme fonksiyonu

list_of_dfs = []

# 720 ay için döngü
for month in range(720):
    def get_values(index, nearest_col, data_dict, num_items=3):
        nearest_str = data.loc[data['hzbnr01'] == index, nearest_col].values
        if nearest_str:
            nearest_list = ast.literal_eval(nearest_str[0])
            values = []
            for i in range(num_items):
                str_index = str(nearest_list[i]) if i < len(nearest_list) else None
                relevant_df = data_dict.get(str_index, pd.DataFrame([[None, None, None]]))
                values.extend(relevant_df.iloc[0, :3].values)
            return values
        return [None] * (3 * num_items)

    # GROUNDWATER
    new_df[['gw_level', 'gw_level_lag_1', 'gw_level_rolling_mean_6_lag_1']] = new_df.index.to_series().map(
        lambda idx: filled_groundwater_dict.get(str(idx), pd.DataFrame([[None, None, None]])).iloc[0, :3].values
    ).apply(pd.Series)

    # GROUNDWATER TEMPERATURE
    new_df[['gw_temp', 'gw_temp_lag_1', 'gw_temp_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_gw_temp', filled_data_gw_temp_dict, num_items=1)
    ).apply(pd.Series)

    # CONDUCTIVITY
    new_df[['conductivity', 'conductivity_lag_1', 'conductivity_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_conductivity', filled_conductivity_dict, num_items=1)
    ).apply(pd.Series)

    # SOURCE FLOW RATE
    new_df[['source_flow_rate', 'source_flow_rate_lag_1', 'source_flow_rate_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_source_fr', filled_source_flow_rate_dict, num_items=1)
    ).apply(pd.Series)

    # SOURCE TEMPERATURE
    new_df[['source_temp', 'source_temp_lag_1', 'source_temp_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_source_temp', filled_source_temp_dict, num_items=1)
    ).apply(pd.Series)

    # SURFACE WATER TEMP
    new_df[['surface_water_temp', 'surface_water_temp_lag_1', 'surface_water_temp_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_owf_temp', filled_surface_water_temp_dict, num_items=1)
    ).apply(pd.Series)

    # SED?MENT
    new_df[['sediment', 'sediment_lag_1', 'sediment_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_sediment', filled_sediment_dict_monthly, num_items=1)
    ).apply(pd.Series)

    # RAIN
    rain_columns = ['first_rain', 'first_rain_lag_1', 'first_rain_rolling_mean_6_lag_1',
                    'second_rain', 'second_rain_lag_1', 'second_rain_rolling_mean_6_lag_1',
                    'third_rain', 'third_rain_lag_1', 'third_rain_rolling_mean_6_lag_1']

    new_df[rain_columns] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_rain', filled_rain_dict, num_items=3)
    ).apply(pd.Series)

    # SNOW
    snow_columns = ['first_snow', 'first_snow_lag_1', 'first_snow_rolling_mean_6_lag_1',
                    'second_snow', 'second_snow_lag_1', 'second_snow_rolling_mean_6_lag_1',
                    'third_snow', 'third_snow_lag_1', 'third_snow_rolling_mean_6_lag_1']

    new_df[snow_columns] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_snow', filled_snow_dict_monthly, num_items=3)
    ).apply(pd.Series)

    # SURFACE WATER LEVEL
    surface_water_level_columns = ['first_surface_water_level', 'first_surface_water_level_lag_1', 'first_surface_water_level_rolling_mean_6_lag_1',
                    'second_surface_water_level', 'second_surface_water_level_lag_1', 'second_surface_water_level_rolling_mean_6_lag_1',
                    'third_surface_water_level', 'third_surface_water_level_lag_1', 'third_surface_water_level_rolling_mean_6_lag_1']

    new_df[surface_water_level_columns] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_owf_level', filled_surface_water_level_dict_monthly, num_items=3)
    ).apply(pd.Series)

    # SURFACE WATER FLOW RATE
    surface_water_fr_columns = ['first_surface_water_fr', 'first_surface_water_fr_lag_1', 'first_surface_water_fr_rolling_mean_6_lag_1',
                    'second_surface_water_fr', 'second_surface_water_fr_lag_1', 'second_surface_water_fr_rolling_mean_6_lag_1',
                    'third_surface_water_fr', 'third_surface_water_fr_lag_1', 'third_surface_water_fr_rolling_mean_6_lag_1']

    new_df[surface_water_fr_columns] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_owf_fr', filled_surface_water_flow_rate_dict_monthly, num_items=3)
    ).apply(pd.Series)




############################################################################################################################

all_dataframes = []

# 720 ay için döngü
for month in range(720):
    new_df = pd.DataFrame()
    def get_values(index, nearest_col, data_dict, num_items=3):
        nearest_str = data.loc[data['hzbnr01'] == index, nearest_col].values
        if nearest_str:
            nearest_list = ast.literal_eval(nearest_str[0])
            values = []
            for i in range(num_items):
                str_index = str(nearest_list[i]) if i < len(nearest_list) else None
                relevant_df = data_dict.get(str_index, pd.DataFrame([[None, None, None]]))
                values.extend(relevant_df.iloc[month, :3].values)
            return values
        return [None] * (3 * num_items)

    # GROUNDWATER
    new_df[['gw_level', 'gw_level_lag_1', 'gw_level_rolling_mean_6_lag_1']] = new_df.index.to_series().map(
        lambda idx: filled_groundwater_dict.get(str(idx), pd.DataFrame([[None, None, None]])).iloc[0, :3].values
    ).apply(pd.Series)

    # GROUNDWATER TEMPERATURE
    new_df[['gw_temp', 'gw_temp_lag_1', 'gw_temp_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_gw_temp', filled_data_gw_temp_dict, num_items=1)
    ).apply(pd.Series)

    # CONDUCTIVITY
    new_df[['conductivity', 'conductivity_lag_1', 'conductivity_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_conductivity', filled_conductivity_dict, num_items=1)
    ).apply(pd.Series)

    # SOURCE FLOW RATE
    new_df[['source_flow_rate', 'source_flow_rate_lag_1', 'source_flow_rate_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_source_fr', filled_source_flow_rate_dict, num_items=1)
    ).apply(pd.Series)

    # SOURCE TEMPERATURE
    new_df[['source_temp', 'source_temp_lag_1', 'source_temp_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_source_temp', filled_source_temp_dict, num_items=1)
    ).apply(pd.Series)

    # SURFACE WATER TEMP
    new_df[['surface_water_temp', 'surface_water_temp_lag_1', 'surface_water_temp_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_owf_temp', filled_surface_water_temp_dict, num_items=1)
    ).apply(pd.Series)

    # SED?MENT
    new_df[['sediment', 'sediment_lag_1', 'sediment_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_sediment', filled_sediment_dict_monthly, num_items=1)
    ).apply(pd.Series)

    # RAIN
    rain_columns = ['first_rain', 'first_rain_lag_1', 'first_rain_rolling_mean_6_lag_1',
                    'second_rain', 'second_rain_lag_1', 'second_rain_rolling_mean_6_lag_1',
                    'third_rain', 'third_rain_lag_1', 'third_rain_rolling_mean_6_lag_1']

    new_df[rain_columns] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_rain', filled_rain_dict, num_items=3)
    ).apply(pd.Series)

    # SNOW
    snow_columns = ['first_snow', 'first_snow_lag_1', 'first_snow_rolling_mean_6_lag_1',
                    'second_snow', 'second_snow_lag_1', 'second_snow_rolling_mean_6_lag_1',
                    'third_snow', 'third_snow_lag_1', 'third_snow_rolling_mean_6_lag_1']

    new_df[snow_columns] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_snow', filled_snow_dict_monthly, num_items=3)
    ).apply(pd.Series)

    # SURFACE WATER LEVEL
    surface_water_level_columns = ['first_surface_water_level', 'first_surface_water_level_lag_1', 'first_surface_water_level_rolling_mean_6_lag_1',
                    'second_surface_water_level', 'second_surface_water_level_lag_1', 'second_surface_water_level_rolling_mean_6_lag_1',
                    'third_surface_water_level', 'third_surface_water_level_lag_1', 'third_surface_water_level_rolling_mean_6_lag_1']

    new_df[surface_water_level_columns] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_owf_level', filled_surface_water_level_dict_monthly, num_items=3)
    ).apply(pd.Series)

    # SURFACE WATER FLOW RATE
    surface_water_fr_columns = ['first_surface_water_fr', 'first_surface_water_fr_lag_1', 'first_surface_water_fr_rolling_mean_6_lag_1',
                    'second_surface_water_fr', 'second_surface_water_fr_lag_1', 'second_surface_water_fr_rolling_mean_6_lag_1',
                    'third_surface_water_fr', 'third_surface_water_fr_lag_1', 'third_surface_water_fr_rolling_mean_6_lag_1']

    new_df[surface_water_fr_columns] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_owf_fr', filled_surface_water_flow_rate_dict_monthly, num_items=3)
    ).apply(pd.Series)

    all_dataframes.append(new_df)


#####################################################3

import numpy as np
import ast
import pandas as pd


def get_values(index, nearest_col, data_dict, num_items=3):
    nearest_str = data.loc[data['hzbnr01'] == index, nearest_col].values

    # Explicitly check if the array is not empty
    if nearest_str.size > 0:
        nearest_list = ast.literal_eval(nearest_str[0])
        values = []
        for i in range(num_items):
            str_index = str(nearest_list[i]) if i < len(nearest_list) else None
            relevant_df = data_dict.get(str_index, pd.DataFrame([[None, None, None]]))
            values.extend(relevant_df.iloc[0, :3].values)
        return values

    return [None] * (3 * num_items)


all_dataframes = []

# 720 ay için döngü
for month in range(720):
    new_df = pd.DataFrame(index=range(len(data)))  # Veri çerçevesinin indeksini belirle

    # GROUNDWATER
    new_df[['gw_level', 'gw_level_lag_1', 'gw_level_rolling_mean_6_lag_1']] = new_df.index.to_series().map(
        lambda idx: filled_groundwater_dict.get(str(idx), pd.DataFrame([[None, None, None]])).iloc[0, :3].values
    ).apply(pd.Series)

    # GROUNDWATER TEMPERATURE
    new_df[['gw_temp', 'gw_temp_lag_1', 'gw_temp_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_gw_temp', filled_data_gw_temp_dict, num_items=1)
    ).apply(pd.Series)

    # CONDUCTIVITY
    new_df[
        ['conductivity', 'conductivity_lag_1', 'conductivity_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_conductivity', filled_conductivity_dict, num_items=1)
    ).apply(pd.Series)

    # SOURCE FLOW RATE
    new_df[['source_flow_rate', 'source_flow_rate_lag_1',
            'source_flow_rate_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_source_fr', filled_source_flow_rate_dict, num_items=1)
    ).apply(pd.Series)

    # SOURCE TEMPERATURE
    new_df[['source_temp', 'source_temp_lag_1', 'source_temp_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_source_temp', filled_source_temp_dict, num_items=1)
    ).apply(pd.Series)

    # SURFACE WATER TEMP
    new_df[['surface_water_temp', 'surface_water_temp_lag_1',
            'surface_water_temp_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_owf_temp', filled_surface_water_temp_dict, num_items=1)
    ).apply(pd.Series)

    # SEDIMENT
    new_df[['sediment', 'sediment_lag_1', 'sediment_rolling_mean_6_lag_1']] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_sediment', filled_sediment_dict_monthly, num_items=1)
    ).apply(pd.Series)

    # RAIN
    rain_columns = ['first_rain', 'first_rain_lag_1', 'first_rain_rolling_mean_6_lag_1',
                    'second_rain', 'second_rain_lag_1', 'second_rain_rolling_mean_6_lag_1',
                    'third_rain', 'third_rain_lag_1', 'third_rain_rolling_mean_6_lag_1']

    new_df[rain_columns] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_rain', filled_rain_dict, num_items=3)
    ).apply(pd.Series)

    # SNOW
    snow_columns = ['first_snow', 'first_snow_lag_1', 'first_snow_rolling_mean_6_lag_1',
                    'second_snow', 'second_snow_lag_1', 'second_snow_rolling_mean_6_lag_1',
                    'third_snow', 'third_snow_lag_1', 'third_snow_rolling_mean_6_lag_1']

    new_df[snow_columns] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_snow', filled_snow_dict_monthly, num_items=3)
    ).apply(pd.Series)

    # SURFACE WATER LEVEL
    surface_water_level_columns = ['first_surface_water_level', 'first_surface_water_level_lag_1',
                                   'first_surface_water_level_rolling_mean_6_lag_1',
                                   'second_surface_water_level', 'second_surface_water_level_lag_1',
                                   'second_surface_water_level_rolling_mean_6_lag_1',
                                   'third_surface_water_level', 'third_surface_water_level_lag_1',
                                   'third_surface_water_level_rolling_mean_6_lag_1']

    new_df[surface_water_level_columns] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_owf_level', filled_surface_water_level_dict_monthly, num_items=3)
    ).apply(pd.Series)

    # SURFACE WATER FLOW RATE
    surface_water_fr_columns = ['first_surface_water_fr', 'first_surface_water_fr_lag_1',
                                'first_surface_water_fr_rolling_mean_6_lag_1',
                                'second_surface_water_fr', 'second_surface_water_fr_lag_1',
                                'second_surface_water_fr_rolling_mean_6_lag_1',
                                'third_surface_water_fr', 'third_surface_water_fr_lag_1',
                                'third_surface_water_fr_rolling_mean_6_lag_1']

    new_df[surface_water_fr_columns] = new_df.index.to_series().apply(
        lambda idx: get_values(idx, 'nearest_owf_fr', filled_surface_water_flow_rate_dict_monthly, num_items=3)
    ).apply(pd.Series)

    all_dataframes.append(new_df)

all_dataframes[0]

# 7.11 -- üsttekinde hata ald?k bugün none'lar?, error'u gpt'ye düzelttirip yeniden çal??t?rd?m
# 20 dk sürdü ve düzelmedi


# main'den getirdiklerim
###################################################
# Birden fazla sözlükten belirli bir ay?n 'val' sütunlar?n? toplamak için bir fonksiyon

def get_monthly_vals(dict_list, year, month):
    monthly_vals = []
    for data_dict in dict_list:
        for df_name, df in data_dict.items():
            df.index = pd.to_datetime(df.index)  # Tarih indeksine sahip oldu?undan emin olun
            for i in df.columns:
                monthly_data = df[(df.index.year == year) & (df.index.month == month)][i]
                monthly_vals.append(monthly_data)
    if monthly_vals:
        return pd.concat(monthly_vals, axis=1)
    else:
        return pd.DataFrame()


# Örne?in, 2023 y?l? Ocak ay? verilerini almak için
monthly_vals = get_monthly_vals(dict_list, 2021, 12)
monthly_vals.shape
#### Normalizasyon yani scaling yapmam?z gerek
####################################################
