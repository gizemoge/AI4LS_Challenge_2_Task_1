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
import pandas as pd
from datetime import datetime
from scipy.spatial import distance
from collections import Counter
import itertools
import pickle
import pandas as pd
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
        df[f'rolling_mean_{window}_lag_{i}'] = df[column_name].shift(i).rolling(window=window).mean()
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


# 720 dataframe
# data isimli DataFrame'in hzbnr01 kolonunu al
index_values = data['hzbnr01']

# hzbnr01 de?erlerini indeks olarak kullanan bo? bir DataFrame olu?tur
new_df = pd.DataFrame(index=index_values)


# todo otomatikle?tirece?imiz yer buras?, ben sadece 1960 ocak'? için yapt?m bunlar?
# GROUNDWATER
# Yeni kolonlar? ba?lat?yoruz
new_df['gw_level'] = None
new_df['gw_level_lag_1'] = None
new_df['gw_level_rolling_mean_6_lag_1'] = None

# Tüm indekslerde dola??yoruz
for index in new_df.index:
    # ?ndeksi string'e çeviriyoruz
    str_index = str(index)

    # E?er sözlükte anahtar mevcutsa, ilgili DataFrame'i al?yoruz
    if str_index in filled_groundwater_dict:
        relevant_df = filled_groundwater_dict[str_index]

        # ?lk sat?rdaki tüm de?erleri al?yoruz
        first_row = relevant_df.iloc[0]

        # Bu de?erleri new_df'deki ilgili sat?rlara ekliyoruz
        new_df.at[index, 'gw_level'] = first_row.iloc[0]  # ?lk kolonun de?eri
        new_df.at[index, 'gw_level_lag_1'] = first_row.iloc[1]  # ?kinci kolonun de?eri
        new_df.at[index, 'gw_level_rolling_mean_6_lag_1'] = first_row.iloc[2]  # Üçüncü kolonun de?eri
    else:
        # Sözlükte anahtar bulunmazsa hata mesaj? yazd?r?yoruz
        print(f"Warning: Key '{str_index}' not found in filled_groundwater_dict")

# GROUNWATER TEMPEERAURE
import ast

# Yeni kolonlar? ba?lat?yoruz
new_df['gw_temp'] = None
new_df['gw_temp_lag_1'] = None
new_df['gw_temp_rolling_mean_6_lag_1'] = None

# Tüm indekslerde dola??yoruz
for index in new_df.index:
    # data DataFrame'inde hzbnr01 ile e?le?en nearest_gw_temp listesini al?yoruz
    nearest_gw_temp_str = data.loc[data['hzbnr01'] == index, 'nearest_gw_temp'].values

    # E?er nearest_gw_temp_str bo? de?ilse
    if len(nearest_gw_temp_str) > 0:
        # nearest_gw_temp listesinin string de?erini gerçek listeye dönü?türüyoruz
        nearest_gw_temp_list = ast.literal_eval(nearest_gw_temp_str[0])  # Stringi listeye dönü?türüyoruz

        # nearest_gw_temp listesinin tek eleman?n? al?yoruz
        if len(nearest_gw_temp_list) > 0:
            str_index = str(nearest_gw_temp_list[0])  # Tek eleman? string'e çeviriyoruz

            # Tek eleman için
            if str_index in filled_data_gw_temp_dict:
                relevant_temp_df = filled_data_gw_temp_dict[str_index]
                if len(relevant_temp_df) > 0:
                    new_df.at[index, 'gw_temp'] = relevant_temp_df.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_temp_df.columns) > 1 and len(relevant_temp_df) > 0:
                    new_df.at[index, 'gw_temp_lag_1'] = relevant_temp_df.iloc[0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_temp_df.columns) > 2 and len(relevant_temp_df) > 0:
                    new_df.at[index, 'gw_temp_rolling_mean_6_lag_1'] = relevant_temp_df.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index}' not found in filled_data_gw_temp_dict")
    else:
        print(f"Warning: No nearest_gw_temp list found for index '{index}'")



# RAIN
# Yeni kolonlar? ba?lat?yoruz
new_df['first_rain'] = None
new_df['first_rain_lag_1'] = None
new_df['first_rain_rolling_mean_6_lag_1'] = None
new_df['second_rain'] = None
new_df['second_rain_lag_1'] = None
new_df['second_rain_rolling_mean_6_lag_1'] = None
new_df['third_rain'] = None
new_df['third_rain_lag_1'] = None
new_df['third_rain_rolling_mean_6_lag_1'] = None

# Tüm indekslerde dola??yoruz
for index in new_df.index:
    # data DataFrame'inde hzbnr01 ile e?le?en nearest_rain listesini al?yoruz
    nearest_rain_str = data.loc[data['hzbnr01'] == index, 'nearest_rain'].values

    # E?er nearest_rain_str bo? de?ilse
    if len(nearest_rain_str) > 0:
        # nearest_rain listesinin string de?erini gerçek listeye dönü?türüyoruz
        nearest_rain_list = ast.literal_eval(nearest_rain_str[0])  # Stringi listeye dönü?türüyoruz

        # nearest_rain listesinin ilk, ikinci ve üçüncü elemanlar?n? al?yoruz
        if len(nearest_rain_list) > 0:
            str_index_first = str(nearest_rain_list[0])  # ?lk eleman? string'e çeviriyoruz
            str_index_second = str(nearest_rain_list[1])  # ?kinci eleman? string'e çeviriyoruz
            str_index_third = str(nearest_rain_list[2])  # Üçüncü eleman? string'e çeviriyoruz

            # ?lk eleman için
            if str_index_first in filled_rain_dict:
                relevant_rain_df_first = filled_rain_dict[str_index_first]
                if len(relevant_rain_df_first) > 0:
                    new_df.at[index, 'first_rain'] = relevant_rain_df_first.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_rain_df_first.columns) > 1 and len(relevant_rain_df_first) > 0:
                    new_df.at[index, 'first_rain_lag_1'] = relevant_rain_df_first.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_rain_df_first.columns) > 2 and len(relevant_rain_df_first) > 0:
                    new_df.at[index, 'first_rain_rolling_mean_6_lag_1'] = relevant_rain_df_first.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index_first}' not found in filled_rain_dict")

            # ?kinci eleman için
            if str_index_second in filled_rain_dict:
                relevant_rain_df_second = filled_rain_dict[str_index_second]
                if len(relevant_rain_df_second) > 0:
                    new_df.at[index, 'second_rain'] = relevant_rain_df_second.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_rain_df_second.columns) > 1 and len(relevant_rain_df_second) > 0:
                    new_df.at[index, 'second_rain_lag_1'] = relevant_rain_df_second.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_rain_df_second.columns) > 2 and len(relevant_rain_df_second) > 0:
                    new_df.at[index, 'second_rain_rolling_mean_6_lag_1'] = relevant_rain_df_second.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index_second}' not found in filled_rain_dict")

            # Üçüncü eleman için
            if str_index_third in filled_rain_dict:
                relevant_rain_df_third = filled_rain_dict[str_index_third]
                if len(relevant_rain_df_third) > 0:
                    new_df.at[index, 'third_rain'] = relevant_rain_df_third.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_rain_df_third.columns) > 1 and len(relevant_rain_df_third) > 0:
                    new_df.at[index, 'third_rain_lag_1'] = relevant_rain_df_third.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_rain_df_third.columns) > 2 and len(relevant_rain_df_third) > 0:
                    new_df.at[index, 'third_rain_rolling_mean_6_lag_1'] = relevant_rain_df_third.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index_third}' not found in filled_rain_dict")
    else:
        print(f"Warning: No nearest_rain list found for index '{index}'")

# SNOW
# Yeni kolonlar? ba?lat?yoruz
new_df['first_snow'] = None
new_df['first_snow_lag_1'] = None
new_df['first_snow_rolling_mean_6_lag_1'] = None
new_df['second_snow'] = None
new_df['second_snow_lag_1'] = None
new_df['second_snow_rolling_mean_6_lag_1'] = None
new_df['third_snow'] = None
new_df['third_snow_lag_1'] = None
new_df['third_snow_rolling_mean_6_lag_1'] = None

# Tüm indekslerde dola??yoruz
for index in new_df.index:
    # data DataFrame'inde hzbnr01 ile e?le?en nearest_snow listesini al?yoruz
    nearest_snow_str = data.loc[data['hzbnr01'] == index, 'nearest_snow'].values

    # E?er nearest_snow_str bo? de?ilse
    if len(nearest_snow_str) > 0:
        # nearest_snow listesinin string de?erini gerçek listeye dönü?türüyoruz
        nearest_snow_list = ast.literal_eval(nearest_snow_str[0])  # Stringi listeye dönü?türüyoruz

        # nearest_snow listesinin ilk, ikinci ve üçüncü elemanlar?n? al?yoruz
        if len(nearest_snow_list) > 0:
            str_index_first = str(nearest_snow_list[0])  # ?lk eleman? string'e çeviriyoruz
            str_index_second = str(nearest_snow_list[1])  # ?kinci eleman? string'e çeviriyoruz
            str_index_third = str(nearest_snow_list[2])  # Üçüncü eleman? string'e çeviriyoruz

            # ?lk eleman için
            if str_index_first in filled_snow_dict_monthly:
                relevant_snow_df_first = filled_snow_dict_monthly[str_index_first]
                if len(relevant_snow_df_first) > 0:
                    new_df.at[index, 'first_snow'] = relevant_snow_df_first.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_snow_df_first.columns) > 1 and len(relevant_snow_df_first) > 0:
                    new_df.at[index, 'first_snow_lag_1'] = relevant_snow_df_first.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_snow_df_first.columns) > 2 and len(relevant_snow_df_first) > 0:
                    new_df.at[index, 'first_snow_rolling_mean_6_lag_1'] = relevant_snow_df_first.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index_first}' not found in filled_snow_dict_monthly")

            # ?kinci eleman için
            if str_index_second in filled_snow_dict_monthly:
                relevant_snow_df_second = filled_snow_dict_monthly[str_index_second]
                if len(relevant_snow_df_second) > 0:
                    new_df.at[index, 'second_snow'] = relevant_snow_df_second.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_snow_df_second.columns) > 1 and len(relevant_snow_df_second) > 0:
                    new_df.at[index, 'second_snow_lag_1'] = relevant_snow_df_second.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_snow_df_second.columns) > 2 and len(relevant_snow_df_second) > 0:
                    new_df.at[index, 'second_snow_rolling_mean_6_lag_1'] = relevant_snow_df_second.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index_second}' not found in filled_snow_dict_monthly")

            # Üçüncü eleman için
            if str_index_third in filled_snow_dict_monthly:
                relevant_snow_df_third = filled_snow_dict_monthly[str_index_third]
                if len(relevant_snow_df_third) > 0:
                    new_df.at[index, 'third_snow'] = relevant_snow_df_third.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_snow_df_third.columns) > 1 and len(relevant_snow_df_third) > 0:
                    new_df.at[index, 'third_snow_lag_1'] = relevant_snow_df_third.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_snow_df_third.columns) > 2 and len(relevant_snow_df_third) > 0:
                    new_df.at[index, 'third_snow_rolling_mean_6_lag_1'] = relevant_snow_df_third.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index_third}' not found in filled_snow_dict_monthly")
    else:
        print(f"Warning: No nearest_snow list found for index '{index}'")

# CONDUCTIVITY
# Yeni kolonlar? ba?lat?yoruz
new_df['conductivity'] = None
new_df['conductivity_lag_1'] = None
new_df['conductivity_rolling_mean_6_lag_1'] = None

# Tüm indekslerde dola??yoruz
for index in new_df.index:
    # data DataFrame'inde hzbnr01 ile e?le?en nearest_conductivity listesini al?yoruz
    nearest_conductivity_str = data.loc[data['hzbnr01'] == index, 'nearest_conductivity'].values

    # E?er nearest_conductivity_str bo? de?ilse
    if len(nearest_conductivity_str) > 0:
        # nearest_conductivity listesinin string de?erini gerçek listeye dönü?türüyoruz
        nearest_conductivity_list = ast.literal_eval(nearest_conductivity_str[0])  # Stringi listeye dönü?türüyoruz

        # nearest_conductivity listesinin tek eleman?n? al?yoruz
        if len(nearest_conductivity_list) > 0:
            str_index = str(nearest_conductivity_list[0])  # Tek eleman? string'e çeviriyoruz

            # Tek eleman için
            if str_index in filled_conductivity_dict:
                relevant_conductivity_df = filled_conductivity_dict[str_index]
                if len(relevant_conductivity_df) > 0:
                    new_df.at[index, 'conductivity'] = relevant_conductivity_df.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_conductivity_df.columns) > 1 and len(relevant_conductivity_df) > 0:
                    new_df.at[index, 'conductivity_lag_1'] = relevant_conductivity_df.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_conductivity_df.columns) > 2 and len(relevant_conductivity_df) > 0:
                    new_df.at[index, 'conductivity_rolling_mean_6_lag_1'] = relevant_conductivity_df.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index}' not found in filled_conductivity_dict")
    else:
        print(f"Warning: No nearest_conductivity list found for index '{index}'")

# SOURCE FLOW RATE
import ast

# Yeni kolonlar? ba?lat?yoruz
new_df['source_flow_rate'] = None
new_df['source_flow_rate_lag_1'] = None
new_df['source_flow_rate_rolling_mean_6_lag_1'] = None

# Tüm indekslerde dola??yoruz
for index in new_df.index:
    # data DataFrame'inde hzbnr01 ile e?le?en nearest_source_fr listesini al?yoruz
    nearest_source_fr_str = data.loc[data['hzbnr01'] == index, 'nearest_source_fr'].values

    # E?er nearest_source_fr_str bo? de?ilse
    if len(nearest_source_fr_str) > 0:
        # nearest_source_fr listesinin string de?erini gerçek listeye dönü?türüyoruz
        nearest_source_fr_list = ast.literal_eval(nearest_source_fr_str[0])  # Stringi listeye dönü?türüyoruz

        # nearest_source_fr listesinin tek eleman?n? al?yoruz
        if len(nearest_source_fr_list) > 0:
            str_index = str(nearest_source_fr_list[0])  # Tek eleman? string'e çeviriyoruz

            # Tek eleman için
            if str_index in filled_source_flow_rate_dict:
                relevant_flow_rate_df = filled_source_flow_rate_dict[str_index]
                if len(relevant_flow_rate_df) > 0:
                    new_df.at[index, 'source_flow_rate'] = relevant_flow_rate_df.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_flow_rate_df.columns) > 1 and len(relevant_flow_rate_df) > 0:
                    new_df.at[index, 'source_flow_rate_lag_1'] = relevant_flow_rate_df.iloc[0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_flow_rate_df.columns) > 2 and len(relevant_flow_rate_df) > 0:
                    new_df.at[index, 'source_flow_rate_rolling_mean_6_lag_1'] = relevant_flow_rate_df.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index}' not found in filled_source_flow_rate_dict")
    else:
        print(f"Warning: No nearest_source_fr list found for index '{index}'")

# SOURCE TEMP
# Yeni kolonlar? ba?lat?yoruz
new_df['source_temp'] = None
new_df['source_temp_lag_1'] = None
new_df['source_temp_rolling_mean_6_lag_1'] = None

# Tüm indekslerde dola??yoruz
for index in new_df.index:
    # data DataFrame'inde hzbnr01 ile e?le?en nearest_source_temp listesini al?yoruz
    nearest_source_temp_str = data.loc[data['hzbnr01'] == index, 'nearest_source_temp'].values

    # E?er nearest_source_temp_str bo? de?ilse
    if len(nearest_source_temp_str) > 0:
        # nearest_source_temp listesinin string de?erini gerçek listeye dönü?türüyoruz
        nearest_source_temp_list = ast.literal_eval(nearest_source_temp_str[0])  # Stringi listeye dönü?türüyoruz

        # nearest_source_temp listesinin tek eleman?n? al?yoruz
        if len(nearest_source_temp_list) > 0:
            str_index = str(nearest_source_temp_list[0])  # Tek eleman? string'e çeviriyoruz

            # Tek eleman için
            if str_index in filled_source_temp_dict:
                relevant_temp_df = filled_source_temp_dict[str_index]
                if len(relevant_temp_df) > 0:
                    new_df.at[index, 'source_temp'] = relevant_temp_df.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_temp_df.columns) > 1 and len(relevant_temp_df) > 0:
                    new_df.at[index, 'source_temp_lag_1'] = relevant_temp_df.iloc[0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_temp_df.columns) > 2 and len(relevant_temp_df) > 0:
                    new_df.at[index, 'source_temp_rolling_mean_6_lag_1'] = relevant_temp_df.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index}' not found in filled_source_temp_dict")
    else:
        print(f"Warning: No nearest_source_temp list found for index '{index}'")

# SURFACE WATER TEMP
# Yeni kolonlar? ba?lat?yoruz
new_df['owf_temp'] = None
new_df['owf_temp_lag_1'] = None
new_df['owf_temp_rolling_mean_6_lag_1'] = None

# Tüm indekslerde dola??yoruz
for index in new_df.index:
    # data DataFrame'inde hzbnr01 ile e?le?en nearest_owf_temp listesini al?yoruz
    nearest_owf_temp_str = data.loc[data['hzbnr01'] == index, 'nearest_owf_temp'].values

    # E?er nearest_owf_temp_str bo? de?ilse
    if len(nearest_owf_temp_str) > 0:
        # nearest_owf_temp listesinin string de?erini gerçek listeye dönü?türüyoruz
        nearest_owf_temp_list = ast.literal_eval(nearest_owf_temp_str[0])  # Stringi listeye dönü?türüyoruz

        # nearest_owf_temp listesinin tek eleman?n? al?yoruz
        if len(nearest_owf_temp_list) > 0:
            str_index = str(nearest_owf_temp_list[0])  # Tek eleman? string'e çeviriyoruz

            # Tek eleman için
            if str_index in filled_surface_water_temp_dict:
                relevant_temp_df = filled_surface_water_temp_dict[str_index]
                if len(relevant_temp_df) > 0:
                    new_df.at[index, 'owf_temp'] = relevant_temp_df.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_temp_df.columns) > 1 and len(relevant_temp_df) > 0:
                    new_df.at[index, 'owf_temp_lag_1'] = relevant_temp_df.iloc[0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_temp_df.columns) > 2 and len(relevant_temp_df) > 0:
                    new_df.at[index, 'owf_temp_rolling_mean_6_lag_1'] = relevant_temp_df.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index}' not found in filled_surface_water_temp_dict")
    else:
        print(f"Warning: No nearest_owf_temp list found for index '{index}'")

# SED?MENT
# Yeni kolonlar? ba?lat?yoruz
new_df['sediment'] = None
new_df['sediment_lag_1'] = None
new_df['sediment_rolling_mean_6_lag_1'] = None

# Tüm indekslerde dola??yoruz
for index in new_df.index:
    # data DataFrame'inde hzbnr01 ile e?le?en nearest_sediment listesini al?yoruz
    nearest_sediment_str = data.loc[data['hzbnr01'] == index, 'nearest_sediment'].values

    # E?er nearest_sediment_str bo? de?ilse
    if len(nearest_sediment_str) > 0:
        # nearest_sediment listesinin string de?erini gerçek listeye dönü?türüyoruz
        nearest_sediment_list = ast.literal_eval(nearest_sediment_str[0])  # Stringi listeye dönü?türüyoruz

        # nearest_sediment listesinin tek eleman?n? al?yoruz
        if len(nearest_sediment_list) > 0:
            str_index = str(nearest_sediment_list[0])  # Tek eleman? string'e çeviriyoruz

            # Tek eleman için
            if str_index in filled_sediment_dict_monthly:
                relevant_sediment_df = filled_sediment_dict_monthly[str_index]
                if len(relevant_sediment_df) > 0:
                    new_df.at[index, 'sediment'] = relevant_sediment_df.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_sediment_df.columns) > 1 and len(relevant_sediment_df) > 0:
                    new_df.at[index, 'sediment_lag_1'] = relevant_sediment_df.iloc[0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_sediment_df.columns) > 2 and len(relevant_sediment_df) > 0:
                    new_df.at[index, 'sediment_rolling_mean_6_lag_1'] = relevant_sediment_df.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index}' not found in filled_sediment_dict_monthly")
    else:
        print(f"Warning: No nearest_sediment list found for index '{index}'")

# SURFACE WATER LEVEL
import ast

# Yeni sütunlar? ekliyoruz
new_df['first_owf_level'] = None
new_df['first_owf_level_lag_1'] = None
new_df['first_owf_level_rolling_mean_6_lag_1'] = None
new_df['second_owf_level'] = None
new_df['second_owf_level_lag_1'] = None
new_df['second_owf_level_rolling_mean_6_lag_1'] = None
new_df['third_owf_level'] = None
new_df['third_owf_level_lag_1'] = None
new_df['third_owf_level_rolling_mean_6_lag_1'] = None

# Tüm indekslerde dola??yoruz
for index in new_df.index:
    # data DataFrame'inde hzbnr01 ile e?le?en nearest_owf_level listesini al?yoruz
    nearest_owf_level_str = data.loc[data['hzbnr01'] == index, 'nearest_owf_level'].values

    # E?er nearest_owf_level_str bo? de?ilse
    if len(nearest_owf_level_str) > 0:
        # nearest_owf_level listesinin string de?erini gerçek listeye dönü?türüyoruz
        nearest_owf_level_list = ast.literal_eval(nearest_owf_level_str[0])  # Stringi listeye dönü?türüyoruz

        # nearest_owf_level listesinin ilk, ikinci ve üçüncü elemanlar?n? al?yoruz
        if len(nearest_owf_level_list) > 0:
            str_index_first = str(nearest_owf_level_list[0])  # ?lk eleman? string'e çeviriyoruz
            str_index_second = str(nearest_owf_level_list[1])  # ?kinci eleman? string'e çeviriyoruz
            str_index_third = str(nearest_owf_level_list[2])  # Üçüncü eleman? string'e çeviriyoruz

            # ?lk eleman için
            if str_index_first in filled_surface_water_level_dict_monthly:
                relevant_owf_level_df_first = filled_surface_water_level_dict_monthly[str_index_first]
                if len(relevant_owf_level_df_first) > 0:
                    new_df.at[index, 'first_owf_level'] = relevant_owf_level_df_first.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_owf_level_df_first.columns) > 1 and len(relevant_owf_level_df_first) > 0:
                    new_df.at[index, 'first_owf_level_lag_1'] = relevant_owf_level_df_first.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_owf_level_df_first.columns) > 2 and len(relevant_owf_level_df_first) > 0:
                    new_df.at[index, 'first_owf_level_rolling_mean_6_lag_1'] = relevant_owf_level_df_first.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index_first}' not found in filled_surface_water_level_dict_monthly")

            # ?kinci eleman için
            if str_index_second in filled_surface_water_level_dict_monthly:
                relevant_owf_level_df_second = filled_surface_water_level_dict_monthly[str_index_second]
                if len(relevant_owf_level_df_second) > 0:
                    new_df.at[index, 'second_owf_level'] = relevant_owf_level_df_second.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_owf_level_df_second.columns) > 1 and len(relevant_owf_level_df_second) > 0:
                    new_df.at[index, 'second_owf_level_lag_1'] = relevant_owf_level_df_second.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_owf_level_df_second.columns) > 2 and len(relevant_owf_level_df_second) > 0:
                    new_df.at[index, 'second_owf_level_rolling_mean_6_lag_1'] = relevant_owf_level_df_second.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index_second}' not found in filled_surface_water_level_dict_monthly")

            # Üçüncü eleman için
            if str_index_third in filled_surface_water_level_dict_monthly:
                relevant_owf_level_df_third = filled_surface_water_level_dict_monthly[str_index_third]
                if len(relevant_owf_level_df_third) > 0:
                    new_df.at[index, 'third_owf_level'] = relevant_owf_level_df_third.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_owf_level_df_third.columns) > 1 and len(relevant_owf_level_df_third) > 0:
                    new_df.at[index, 'third_owf_level_lag_1'] = relevant_owf_level_df_third.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_owf_level_df_third.columns) > 2 and len(relevant_owf_level_df_third) > 0:
                    new_df.at[index, 'third_owf_level_rolling_mean_6_lag_1'] = relevant_owf_level_df_third.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index_third}' not found in filled_surface_water_level_dict_monthly")
    else:
        print(f"Warning: No nearest_owf_level list found for index '{index}'")

# SURFACE WATER FLOW RATE
# Yeni sütunlar? ekliyoruz
new_df['first_owf_fr'] = None
new_df['first_owf_fr_lag_1'] = None
new_df['first_owf_fr_rolling_mean_6_lag_1'] = None
new_df['second_owf_fr'] = None
new_df['second_owf_fr_lag_1'] = None
new_df['second_owf_fr_rolling_mean_6_lag_1'] = None
new_df['third_owf_fr'] = None
new_df['third_owf_fr_lag_1'] = None
new_df['third_owf_fr_rolling_mean_6_lag_1'] = None

# Tüm indekslerde dola??yoruz
for index in new_df.index:
    # data DataFrame'inde hzbnr01 ile e?le?en nearest_owf_fr listesini al?yoruz
    nearest_owf_fr_str = data.loc[data['hzbnr01'] == index, 'nearest_owf_fr'].values

    # E?er nearest_owf_fr_str bo? de?ilse
    if len(nearest_owf_fr_str) > 0:
        # nearest_owf_fr listesinin string de?erini gerçek listeye dönü?türüyoruz
        nearest_owf_fr_list = ast.literal_eval(nearest_owf_fr_str[0])  # Stringi listeye dönü?türüyoruz

        # nearest_owf_fr listesinin ilk, ikinci ve üçüncü elemanlar?n? al?yoruz
        if len(nearest_owf_fr_list) > 0:
            str_index_first = str(nearest_owf_fr_list[0])  # ?lk eleman? string'e çeviriyoruz
            str_index_second = str(nearest_owf_fr_list[1])  # ?kinci eleman? string'e çeviriyoruz
            str_index_third = str(nearest_owf_fr_list[2])  # Üçüncü eleman? string'e çeviriyoruz

            # ?lk eleman için
            if str_index_first in filled_surface_water_flow_rate_dict_monthly:
                relevant_owf_fr_df_first = filled_surface_water_flow_rate_dict_monthly[str_index_first]
                if len(relevant_owf_fr_df_first) > 0:
                    new_df.at[index, 'first_owf_fr'] = relevant_owf_fr_df_first.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_owf_fr_df_first.columns) > 1 and len(relevant_owf_fr_df_first) > 0:
                    new_df.at[index, 'first_owf_fr_lag_1'] = relevant_owf_fr_df_first.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_owf_fr_df_first.columns) > 2 and len(relevant_owf_fr_df_first) > 0:
                    new_df.at[index, 'first_owf_fr_rolling_mean_6_lag_1'] = relevant_owf_fr_df_first.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index_first}' not found in filled_surface_water_flow_rate_dict_monthly")

            # ?kinci eleman için
            if str_index_second in filled_surface_water_flow_rate_dict_monthly:
                relevant_owf_fr_df_second = filled_surface_water_flow_rate_dict_monthly[str_index_second]
                if len(relevant_owf_fr_df_second) > 0:
                    new_df.at[index, 'second_owf_fr'] = relevant_owf_fr_df_second.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_owf_fr_df_second.columns) > 1 and len(relevant_owf_fr_df_second) > 0:
                    new_df.at[index, 'second_owf_fr_lag_1'] = relevant_owf_fr_df_second.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_owf_fr_df_second.columns) > 2 and len(relevant_owf_fr_df_second) > 0:
                    new_df.at[index, 'second_owf_fr_rolling_mean_6_lag_1'] = relevant_owf_fr_df_second.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index_second}' not found in filled_surface_water_flow_rate_dict_monthly")

            # Üçüncü eleman için
            if str_index_third in filled_surface_water_flow_rate_dict_monthly:
                relevant_owf_fr_df_third = filled_surface_water_flow_rate_dict_monthly[str_index_third]
                if len(relevant_owf_fr_df_third) > 0:
                    new_df.at[index, 'third_owf_fr'] = relevant_owf_fr_df_third.iloc[0, 0]  # ?lk sat?r?n ilk kolon de?eri
                if len(relevant_owf_fr_df_third.columns) > 1 and len(relevant_owf_fr_df_third) > 0:
                    new_df.at[index, 'third_owf_fr_lag_1'] = relevant_owf_fr_df_third.iloc[
                        0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                if len(relevant_owf_fr_df_third.columns) > 2 and len(relevant_owf_fr_df_third) > 0:
                    new_df.at[index, 'third_owf_fr_rolling_mean_6_lag_1'] = relevant_owf_fr_df_third.iloc[
                        0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
            else:
                print(f"Warning: Key '{str_index_third}' not found in filled_surface_water_flow_rate_dict_monthly")
    else:
        print(f"Warning: No nearest_owf_fr list found for index '{index}'")





# Bo? bir liste olu?turuyoruz
list_of_dfs = []

# 720 ay için döngü
for month in range(720):
    # ?lgili index de?erlerini elde ediyoruz
    index_values = data['hzbnr01']

    # Yeni bir DataFrame olu?turuyoruz
    new_df = pd.DataFrame(index=index_values)

    # GROUNDWATER
    new_df['gw_level'] = None
    new_df['gw_level_lag_1'] = None
    new_df['gw_level_rolling_mean_6_lag_1'] = None

    for index in new_df.index:
        str_index = str(index)
        if str_index in filled_groundwater_dict:
            relevant_df = filled_groundwater_dict[str_index]
            if len(relevant_df) > month:
                row = relevant_df.iloc[month]
                new_df.at[index, 'gw_level'] = row[0]
                new_df.at[index, 'gw_level_lag_1'] = row[1]
                new_df.at[index, 'gw_level_rolling_mean_6_lag_1'] = row[2]
        else:
            print(f"Warning: Key '{str_index}' not found in filled_groundwater_dict")

    # GROUNDWATER TEMPERATURE
    new_df['gw_temp'] = None
    new_df['gw_temp_lag_1'] = None
    new_df['gw_temp_rolling_mean_6_lag_1'] = None

    for index in new_df.index:
        nearest_gw_temp_str = data.loc[data['hzbnr01'] == index, 'nearest_gw_temp'].values
        if len(nearest_gw_temp_str) > 0:
            nearest_gw_temp_list = ast.literal_eval(nearest_gw_temp_str[0])
            if len(nearest_gw_temp_list) > 0:
                str_index = str(nearest_gw_temp_list[0])
                if str_index in filled_data_gw_temp_dict:
                    relevant_temp_df = filled_data_gw_temp_dict[str_index]
                    if len(relevant_temp_df) > month:
                        row = relevant_temp_df.iloc[month]
                        new_df.at[index, 'gw_temp'] = row[0]
                        new_df.at[index, 'gw_temp_lag_1'] = row[1]
                        new_df.at[index, 'gw_temp_rolling_mean_6_lag_1'] = row[2]
                else:
                    print(f"Warning: Key '{str_index}' not found in filled_data_gw_temp_dict")
        else:
            print(f"Warning: No nearest_gw_temp list found for index '{index}'")

    # RAIN
    new_df['first_rain'] = None
    new_df['first_rain_lag_1'] = None
    new_df['first_rain_rolling_mean_6_lag_1'] = None
    new_df['second_rain'] = None
    new_df['second_rain_lag_1'] = None
    new_df['second_rain_rolling_mean_6_lag_1'] = None
    new_df['third_rain'] = None
    new_df['third_rain_lag_1'] = None
    new_df['third_rain_rolling_mean_6_lag_1'] = None

    for index in new_df.index:
        nearest_rain_str = data.loc[data['hzbnr01'] == index, 'nearest_rain'].values
        if len(nearest_rain_str) > 0:
            nearest_rain_list = ast.literal_eval(nearest_rain_str[0])
            if len(nearest_rain_list) > 0:
                for i, rain_index in enumerate(nearest_rain_list[:3]):  # ?lk üç eleman? al?yoruz
                    str_index = str(rain_index)
                    if str_index in filled_rain_dict:
                        relevant_rain_df = filled_rain_dict[str_index]
                        if len(relevant_rain_df) > month:
                            row = relevant_rain_df.iloc[month]
                            new_df.at[index, f'{["first", "second", "third"][i]}_rain'] = row[0]
                            new_df.at[index, f'{["first", "second", "third"][i]}_rain_lag_1'] = row[1]
                            new_df.at[index, f'{["first", "second", "third"][i]}_rain_rolling_mean_6_lag_1'] = row[2]
                    else:
                        print(f"Warning: Key '{str_index}' not found in filled_rain_dict")
        else:
            print(f"Warning: No nearest_rain list found for index '{index}'")



    # Listeye ekliyoruz
    list_of_dfs.append(new_df)

# Art?k list_of_dfs içinde 720 adet DataFrame bulunuyor

len(list_of_dfs)

# model
# 1. Veri Haz?rl???
# DataFrame'leri numpy array'lerine dönü?türüp birle?tirin
data = np.array([df.values for df in list_of_dfs])
data = data.astype(np.float32)  # Verilerin float32 tipinde oldu?undan emin olun
print(data.shape)  # (720, 487, 15)

# 2. Pencereleme
def create_windows(data, window_size, forecast_horizon):
    X, y = [], []
    num_time_steps = data.shape[0]
    num_rows = data.shape[1]
    num_features = data.shape[2]

    for start in range(num_time_steps - window_size - forecast_horizon + 1):
        end = start + window_size
        X.append(data[start:end, :, :])  # (window_size, num_rows, num_features)
        y.append(data[end:end + forecast_horizon, :, :])  # (forecast_horizon, num_rows, num_features)

    return np.array(X).astype(np.float32), np.array(y).astype(np.float32)

window_size = 12  # Geçmi? veriler için pencere boyutu
forecast_horizon = 26  # Gelecekteki ad?m say?s? (26 ad?m tahmin edilecek)
X, y = create_windows(data, window_size, forecast_horizon)
print(X.shape)  # (672, 12, 487, 15)
print(y.shape)  # (672, 26, 487, 15)

# 3. E?itim ve Test Setlerine Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. LSTM Modelini Olu?turma ve E?itim
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, data.shape[1] * data.shape[2])))  # (window_size, num_rows * num_features)
model.add(LSTM(units=50, return_sequences=True))
model.add(Dense(forecast_horizon * data.shape[1] * data.shape[2]))  # Ç?k?? katman?, tahmin edilmesi gereken ad?m say?s?na göre ayarlanmal?
model.compile(optimizer='adam', loss='mse')

# Modeli e?itim
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
