import pickle

with open("pkl_files/grace_imputed_in_dict.pkl", "rb") as f:
    grace_dict = pickle.load(f)


for key, value in grace_dict.items():
    print(value["lwe_thickness"].value_counts())



for key, value in grace_dict.items():
    value_counts = value["lwe_thickness"].value_counts()

    # Ortalamay? ve medyan? hesapla
    mean_value = value_counts.mean()
    median_value = value_counts.median()

    # Sonuçlar? yazd?r
    print(f"DataFrame: {key}")
    print(f"Ortalama: {mean_value}")
    print(f"Medyan: {median_value}")
    print("-----------")

#------------------------------------

# Merging Gldas and GRACE datasets
with open("pkl_files/gldas_dict_2010_2024.pkl", "rb") as f:
    gldas_dict_2010_2024 = pickle.load(f)

with open("pkl_files/grace_imputed_in_dict.pkl", "rb") as f:
    grace_dict = pickle.load(f)

with open("pkl_files/gldas_dict_2004_2009.pkl", "rb") as f:
    gldas_dict_2004_2009 = pickle.load(f)

with open('pkl_files/intersection_set.pkl', 'rb') as file:
    intersection_set = pickle.load(file)

gldas_dict_2010_2024.pop('202405', None)

# Merging supplemental_material_for_task_2 and Gldas
for key in gldas_dict_2010_2024.keys():

    gldas_df = gldas_dict_2010_2024[key]
    grace_df = grace_dict[key]

    if grace_df is not None:
        merged_df = gldas_df.merge(grace_df[['lat', 'lon', 'lwe_thickness']], on=['lat', 'lon'], how='inner')

        gldas_dict_2010_2024[key] = merged_df


filtered_dict = {}
for key, df in gldas_dict_2004_2009.items():

    filtered_df = df[df[['lat', 'lon']].apply(tuple, axis=1).isin(intersection_set)]
    filtered_dict[key] = filtered_df

gldas_dict_2004_2009 = filtered_dict.copy()


# Selecting coordinates in every 209 given longitude
def reduce_to_first_of_19(df):
    return df.iloc[::19, :]


def convert_cols(df, input_col):
    col_type = input_col.split("_")[-1]

    if col_type == "tavg":
        df[f"new_{input_col}"] = df[input_col] * 10800 * 8 * 30
        df.drop(input_col, axis=1, inplace=True)

    elif col_type == "acc":
        df[f"new_{input_col}"] = df[input_col] * 8 * 30
        df.drop(input_col, axis=1, inplace=True)


def process_data(dict):
    results_dict = {}

    for key, df in dict.items():
        results_dict[key] = reduce_to_first_of_19(df)
        results_dict[key].reset_index(drop=True, inplace=True)

    for month, df in results_dict.items():
        for col in df.columns:
            if "_tavg" in col or "_acc" in col:
                convert_cols(df, col)

        try:
            df['MSW'] = (df['new_Rainf_f_tavg'] + df['new_Qsb_acc']) - (
                        df['new_Evap_tavg'] - df['new_ESoil_tavg'] + df['new_Qs_acc'])
        except KeyError as e:
            print(f"KeyError: {e}. Bu sütun eksik olabilir.")

        df.rename(columns={'SWE_inst': 'MSN'}, inplace=True)

        # 'lwe_thickness' sütunu varsa deltaTWS hesaplan?yor
        if 'lwe_thickness' in df.columns:
            df['deltaTWS'] = df["lwe_thickness"] * 10.25

        df['MSM'] = (df["SoilMoi0_10cm_inst"] + df["SoilMoi10_40cm_inst"] + df["SoilMoi40_100cm_inst"] +
                     df["SoilMoi100_200cm_inst"])

        df['SoilTMP0_avg'] = (df['SoilTMP0_10cm_inst'] + df['SoilTMP10_40cm_inst'] + df['SoilTMP40_100cm_inst'] +
                              df['SoilTMP100_200cm_inst'])

        # Silinecek kolonlar listesi
        cols_to_drop = ['SoilMoi0_10cm_inst', 'SoilMoi10_40cm_inst', 'SoilMoi40_100cm_inst', 'SoilMoi100_200cm_inst',
                        'SoilTMP0_10cm_inst', 'SoilTMP10_40cm_inst', 'SoilTMP40_100cm_inst', 'SoilTMP100_200cm_inst']

        # 'lwe_thickness' varsa, silinecek kolonlar listesine ekleniyor
        if 'lwe_thickness' in df.columns:
            cols_to_drop.append('lwe_thickness')

        df.drop([col for col in cols_to_drop if col in df.columns], axis=1, inplace=True)

        results_dict[month] = df

    return results_dict


results_dict_19_2010_2024 = process_data(gldas_dict_2010_2024)
results_dict_19_2004_2009 = process_data(gldas_dict_2004_2009)



# gldas2004-2009 daki 'MSW', 'MSM', 'MSN ortalamalar?n? hesapla:

# ?lk DataFrame'i al
first_df = next(iter(results_dict_19_2004_2009.values()))

# Lat ve Lon kolonlar?n? al
lat_lon_pairs = first_df[['lat', 'lon']].copy()

# Tüm DataFrame'lerdeki lat ve lon kolonlar?n? kar??la?t?r
for key, df in results_dict_19_2004_2009.items():
    if not df[['lat', 'lon']].equals(lat_lon_pairs):
        print(f"{key} DataFrame'inde lat ve lon kolonlar? farkl?.")
        break
else:
    print("Tüm DataFrame'lerde lat ve lon kolonlar? ayn?.")
# ayn?lar




# ?lk DataFrame'den lat ve lon kolonlar?n? al
first_df = next(iter(results_dict_19_2004_2009.values()))
mean_df = first_df[['lat', 'lon']].copy()

# MSN_mean, MSW_mean ve MSM_mean kolonlar?n? ekle
mean_df[['MSN_mean', 'MSW_mean', 'MSM_mean']] = 0.0

# Her ölçüm noktas? için 72 ayl?k ortalamalar? hesapla
for index in range(len(first_df)):
    mean_df.loc[index, ['MSN_mean', 'MSW_mean', 'MSM_mean']] = [
        sum(df[col].iloc[index] for df in results_dict_19_2004_2009.values()) / len(results_dict_19_2004_2009)
        for col in ['MSN', 'MSW', 'MSM']]


"""
###################33 kontrol - ilk sat?ra bak?yorum
# import pandas as pd
# 
# # Bo? bir liste olu?turun
# first_rows = []
# 
# # Sözlükteki her bir dataframe'in ilk sat?r?n? al?n
# for key in sorted(results_dict_19_2004_2009.keys()):
#     df = results_dict_19_2004_2009[key]
#     first_rows.append(df.iloc[0])  # ?lk sat?r? ekleyin
# 
# # ?lk sat?rlardan olu?an bir dataframe olu?turun
# first_rows_df = pd.DataFrame(first_rows)
# 
# # ?lgili sütunlar?n ortalamalar?n? hesaplay?n
# averages = first_rows_df[['MSN', 'MSW', 'MSM']].mean()
# 
# # Sonuçlar? görüntüleyin
# print(averages)

######################## olmus
"""

# results_dict_2010_2024 sözlü?üne deltal? de?erleri getirme

# Her bir ay için sözlükteki dataframe'lerde mean_df'den fark hesaplayarak yeni kolonlar ekleme
for key, df in results_dict_19_2010_2024.items():
    # mean_df ile ayn? lat-lon e?le?melerini buluyoruz
    df_merged = df.merge(mean_df[['lat', 'lon', 'MSN_mean', 'MSW_mean', 'MSM_mean']], on=['lat', 'lon'])

    # Yeni kolonlar? olu?turuyoruz
    df['delta_MSN'] = df['MSN'] - df_merged['MSN_mean']
    df['delta_MSW'] = df['MSW'] - df_merged['MSW_mean']
    df['delta_MSM'] = df['MSM'] - df_merged['MSM_mean']

    # Güncellenmi? dataframe'i sözlü?e tekrar kaydediyoruz
    results_dict_19_2010_2024[key] = df


# delta groundwater'? getiriyoruz ?imdi
#        deltaMGw = deltaTWS - deltaMSM - deltaMSN - deltaMSw

for key, df in results_dict_19_2010_2024.items():
    df['delta_MGW'] = df['deltaTWS'] - df['delta_MSM'] - df['delta_MSN']  - df['delta_MSW']



# with open('supplemental_material_for_task_2/pkl_files/results_dict_19_2004_2009.pkl', 'wb') as file:
#     pickle.dump(results_dict_19_2004_2009, file)
#
# with open('supplemental_material_for_task_2/pkl_files/results_dict_19_2010_2024.pkl', 'wb') as file:
#     pickle.dump(results_dict_19_2010_2024, file)





data_dict = results_dict_19_2010_2024.copy()


#### model
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# E?itim seti (örne?in 201001'den 201812'ye kadar)
train_dfs = [df for key, df in data_dict.items() if 201001 <= int(key) <= 201812]
train_data = pd.concat(train_dfs)

# Test seti (örne?in 201901'den 202404'e kadar)
test_dfs = [df for key, df in data_dict.items() if 201901 <= int(key) <= 202404]
test_data = pd.concat(test_dfs)




# SMAPE (Symmetric Mean Absolute Percentage Error) hesaplama
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


# Tüm koordinatlar için özellik önemlerini, SMAPE'yi, tahmin ve gerçek sonuçlar?n? saklayaca??z
all_feature_importances = {}
all_smape_scores = {}
all_predictions = {}
all_true_values = {}  # Gerçek de?erleri saklamak için yeni bir sözlük

# Her bir koordinat için i?lem yapaca??z
coordinates = train_data[['lat', 'lon']].drop_duplicates()

# Ölçekleyici olu?tur
scaler_X = StandardScaler()
scaler_y = StandardScaler()

for coord in coordinates.itertuples(index=False):
    lat, lon = coord

    # Bu koordinata ait veriyi seçiyoruz
    coord_train_data = train_data[(train_data['lat'] == lat) & (train_data['lon'] == lon)]
    coord_test_data = test_data[(test_data['lat'] == lat) & (test_data['lon'] == lon)]

    if len(coord_train_data) == 0 or len(coord_test_data) == 0:
        continue

    # E?itim ve hedef ayr?m? (e?itim seti)
    X_train = coord_train_data.drop(columns=['delta_MGW'])
    y_train = coord_train_data['delta_MGW'].values.reshape(-1, 1)

    # Test seti
    X_test = coord_test_data.drop(columns=['delta_MGW'])
    y_test = coord_test_data['delta_MGW'].values.reshape(-1, 1)

    # E?itim ve test verilerini ölçeklendirme
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    X_test_scaled = scaler_X.transform(X_test)

    # Model (Random Forest)
    model = RandomForestRegressor()
    model.fit(X_train_scaled, y_train_scaled.ravel())

    # Tahminler
    y_pred_scaled = model.predict(X_test_scaled)

    # Tahminlerin ters ölçeklendirilmesi (inverse scaling)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

    # Özellik önemi
    feature_importance = model.feature_importances_

    # SMAPE skoru
    smape_score = smape(y_test, y_pred)

    # Sonuçlar? sakla
    all_feature_importances[(lat, lon)] = feature_importance
    all_smape_scores[(lat, lon)] = smape_score
    all_predictions[(lat, lon)] = y_pred  # Tahminleri sakla
    all_true_values[(lat, lon)] = y_test  # Gerçek de?erleri sakla

# Ortalamas?n? hesapla
average_smape = np.mean(list(all_smape_scores.values()))
print(f'Ortalama SMAPE: {average_smape:.2f}%')



listo = sorted(all_smape_scores.values(), reverse=True)

listo[:100]