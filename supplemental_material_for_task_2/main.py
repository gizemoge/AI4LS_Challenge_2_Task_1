import pandas as pd
import xarray as xr
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

########################################################################################################################
# GRACE
########################################################################################################################
# Opening .nc data files and converting the variables from these files into dataframes
df_land = xr.open_dataset('supplemental_material_for_task_2/datasets/CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc')
df_land = df_land['LO_val'].to_dataframe().reset_index()

df_lwe = xr.open_dataset('supplemental_material_for_task_2/datasets/CSR_GRACE_GRACE-FO_RL0602_Mascons_all-corrections.nc')
df_lwe = df_lwe["lwe_thickness"].to_dataframe().reset_index()

# Reducing the df_lwe dataframe based on the land mask
df_land_expanded = pd.concat([df_land['LO_val']] * 232, ignore_index=True)

df = pd.concat([df_lwe, df_land_expanded], axis=1)
df = df[df['LO_val'] == 1]
df.drop("LO_val", axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)

# Transforming "time" column to date format
missing_months = ('2002-06;2002-07;2003-06;2011-01;2011-06;2012-05;2012-10;2013-03;2013-08;2013-09;2014-02;2014-07;'
                  '2014-12;2015-06;2015-10;2015-11;2016-04;2016-09;2016-10;2017-02;2017-02;2017-07;2017-08;2017-09;'
                  '2017-10;2017-11;2017-12;2018-01;2018-02;2018-03;2018-04;2018-05;2018-08;2018-09')
missing_months_list = missing_months.split(';')

start_date = pd.Period('2002-04', freq='M')
all_months = pd.period_range(start=start_date, end='2024-04', freq='M')

filtered_months = [str(month) for month in all_months if str(month) not in missing_months_list]

df['time'] = pd.Series(filtered_months).repeat(df.shape[0] // len(filtered_months)).reset_index(drop=True)[:df.shape[0]]

# GRACE after 2010.01 in time column
df = df[df['time'] >= '2010-01']

df['lon'] = df['lon'].apply(lambda x: x - 360 if x > 180 else x)

df.reset_index(drop=True, inplace=True)

# Checking whether coordinates are same for each month
reference_month_grace = df[df['time'] == '2010-01'][['lat', 'lon']].reset_index(drop=True)

comparison = {}
for month in df['time'].unique():
    if month != '2010-01':
        comparison[month] = df[df['time'] == month][['lat', 'lon']].reset_index(drop=True)

results = {month: (reference_month_grace.equals(data) for month, data in comparison.items())}

for month, is_equal in results.items():
    print(f"Comparison result for {month}: {'Same' if is_equal else 'Different'}")

########################################################################################################################
# GLDAS
########################################################################################################################
with open('supplemental_material_for_task_2/pkl_files/gldas_dict_2010_2024.pkl', 'rb') as file:
    gldas_dict_2010_2024 = pickle.load(file)

# Checking whether coordinates are same for each month
coordinates_per_df_gldas = [set(zip(df['lat'], df['lon'])) for df in gldas_dict_2010_2024.values()]
all_same = all(coords == coordinates_per_df_gldas[0] for coords in coordinates_per_df_gldas)
if all_same:
    print("Yes")
else:
    print("No")

########################################################################################################################
# Further Feature Engineering
########################################################################################################################
# Intersection of latitude and longitude couples that come from Gldas and GRACE datasets.
a_month_set = set(zip(reference_month_grace['lat'], reference_month_grace['lon']))
coordinates_set = set(coordinates_per_df_gldas[0])
intersection_set = a_month_set.intersection(coordinates_set)

# Editing the coordinates in GLDAS according to the intersection set.
filtered_dfs = {}

for key, dataframe in gldas_dict_2010_2024.items():
    dataframe['coord_tuple'] = list(zip(dataframe['lat'], dataframe['lon']))

    filtered_df = dataframe[dataframe['coord_tuple'].apply(lambda x: x in intersection_set)]

    filtered_dfs[key] = filtered_df.drop(columns=['coord_tuple'])

for key, dataframe in filtered_dfs.items():
    dataframe.reset_index(drop=True, inplace=True)

# Editing the coordinates in GRACE according to the intersection set
gizmo = df.groupby('time', group_keys=False).apply(lambda group: group[group[['lat', 'lon']].apply(tuple, axis=1).isin(intersection_set)])

df.reset_index(drop=True, inplace=True)


# with open("supplemental_material_for_task_2/pkl_files/df.pkl", "wb") as f:
#     pickle.dump(df, f)

# with open('supplemental_material_for_task_2/pkl_files/df.pkl', 'rb') as file:
#     df = pickle.load(file)


# Filling in the missing months in the dataframe and assigning nan values to the lew_thickness columns.
df['time'] = pd.to_datetime(df['time'])
reference_lat_lon = df[df['time'] == '2010-01-01'][['lat', 'lon']].drop_duplicates()

existing_months = df['time'].drop_duplicates()
all_months = pd.date_range(start='2010-01-01', end='2024-04-01', freq='MS')
missing_months = all_months.difference(existing_months)

missing_data = pd.concat(
    [pd.DataFrame({
        'time': [month] * len(reference_lat_lon),
        'lat': reference_lat_lon['lat'].values,
        'lon': reference_lat_lon['lon'].values,
        'lwe_thickness': np.nan})
     for month in missing_months]
)

df_filled_corrected = pd.concat([df, missing_data]).drop_duplicates(subset=['time', 'lat', 'lon']).sort_values(by=['time', 'lat', 'lon']).reset_index(drop=True)



# GRACE: dataframe to dictionary
df_filled_corrected['key'] = df_filled_corrected['time'].dt.strftime('%Y%m')

result_dict = {key: group.drop(columns='key') for key, group in df_filled_corrected.groupby('key')}

for key, value in result_dict.items():
    value.reset_index(inplace=True, drop=True)


# Imputing NaN Values in the dictionary.
for month_key, month_df in result_dict.items():
    current_month = month_key[-2:]
    measurement_index = month_df.index

    for i in measurement_index:
        if pd.isna(month_df.at[i, 'lwe_thickness']):
            other_year_values = []
            for year in range(2010, 2025):
                year_key = f"{year}{current_month}"
                if year_key in result_dict:
                    other_year_df = result_dict[year_key]
                    if i < len(other_year_df):
                        value = other_year_df.at[i, 'lwe_thickness']
                        if pd.notna(value):
                            other_year_values.append(value)

            if other_year_values:
                average_value = np.mean(other_year_values)
                month_df.at[i, 'lwe_thickness'] = average_value


# Creating .pkl file of the dictionary
# with open('supplemental_material_for_task_2/pkl_files/grace_imputed_in_dict.pkl', 'wb') as f:
#     pickle.dump(result_dict, f)


# Merging Gldas and GRACE datasets
with open("supplemental_material_for_task_2/pkl_files/gldas_dict_2010_2024.pkl", "rb") as f:
    gldas_dict_2010_2024 = pickle.load(f)

with open("supplemental_material_for_task_2/pkl_files/grace_imputed_in_dict.pkl", "rb") as f:
    grace_dict = pickle.load(f)

with open("supplemental_material_for_task_2/pkl_files/gldas_dict_2004_2009.pkl", "rb") as f:
    gldas_dict_2004_2009 = pickle.load(f)

with open('supplemental_material_for_task_2/pkl_files/intersection_set.pkl', 'rb') as file:
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
def reduce_to_first_of_209(df):
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
        results_dict[key] = reduce_to_first_of_209(df)
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
            df['deltaTWS'] = df["lwe_thickness"] * 10

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


results_dict_2010_2024 = process_data(gldas_dict_2010_2024)
results_dict_2004_2009 = process_data(gldas_dict_2004_2009)


# Calculating the averages of 'MSW', 'MSM', 'MSN in results_dict_2004_2009.
first_df = next(iter(results_dict_2004_2009.values()))

lat_lon_pairs = first_df[['lat', 'lon']].copy()


for key, df in results_dict_2004_2009.items():
    if not df[['lat', 'lon']].equals(lat_lon_pairs):
        print(f"{key} lat and lon columns are different.")
        break
else:
    print("All DataFrames have the same lat and lon columns.")


first_df = next(iter(results_dict_2004_2009.values()))
mean_df = first_df[['lat', 'lon']].copy()

mean_df[['MSN_mean', 'MSW_mean', 'MSM_mean']] = 0.0

# Her ölçüm noktas? için 72 ayl?k ortalamalar? hesapla
for index in range(len(first_df)):
    mean_df.loc[index, ['MSN_mean', 'MSW_mean', 'MSM_mean']] = [
        sum(df[col].iloc[index] for df in results_dict_2004_2009.values()) / len(results_dict_2004_2009)
        for col in ['MSN', 'MSW', 'MSM']]


first_rows = []

for key in sorted(results_dict_2004_2009.keys()):
    df = results_dict_2004_2009[key]
    first_rows.append(df.iloc[0])

first_rows_df = pd.DataFrame(first_rows)

averages = first_rows_df[['MSN', 'MSW', 'MSM']].mean()


for key, df in results_dict_2010_2024.items():
    df_merged = df.merge(mean_df[['lat', 'lon', 'MSN_mean', 'MSW_mean', 'MSM_mean']], on=['lat', 'lon'])

    # Yeni kolonlar? olu?turuyoruz
    df['delta_MSN'] = df['MSN'] - df_merged['MSN_mean']
    df['delta_MSW'] = df['MSW'] - df_merged['MSW_mean']
    df['delta_MSM'] = df['MSM'] - df_merged['MSM_mean']

    results_dict_2010_2024[key] = df


#        deltaMGw = deltaTWS - deltaMSM - deltaMSN - deltaMSw
for key, df in results_dict_2010_2024.items():
    df['delta_MGW'] = df['deltaTWS'] - df['delta_MSM'] - df['delta_MSN']  - df['delta_MSW']


# with open('supplemental_material_for_task_2/pkl_files/new_12661_results_dict_2004_2009.pkl', 'wb') as file:
#     pickle.dump(results_dict_2004_2009, file)
#
# with open('supplemental_material_for_task_2/pkl_files/new_12661_results_dict_2010_2024.pkl', 'wb') as file:
#     pickle.dump(results_dict_2010_2024, file)


# Train and test split
train_dict = {}
test_dict = {}

for key, df in results_dict_2010_2024.items():
    date_key = pd.to_datetime(str(key), format='%Y%m')
    year = date_key.year

    if 2010 <= year <= 2018:
        train_dict[key] = df
    elif 2019 <= year <= 2024:
        test_dict[key] = df


# with open('supplemental_material_for_task_2/pkl_files/new_1151_train_dict.pkl', 'wb') as file:
#     pickle.dump(train_dict, file)
#
# with open('supplemental_material_for_task_2/pkl_files/new_1151_test_dict.pkl', 'wb') as file:
#     pickle.dump(test_dict, file)
#
# with open('supplemental_material_for_task_2/pkl_files/train_dict.pkl', 'rb') as file:
#     train_dict = pickle.load(file)
#
# with open('supplemental_material_for_task_2/pkl_files/test_dict.pkl', 'rb') as file:
#     test_dict = pickle.load(file)


# Troubleshooting:
# Whether the 'lat' and 'lon' columns of DataFrames in the dictionary are different or the same,
# and prints which DataFrames have different latitude values or the same longitude values.
lat_values = [df['lat'].values for df in test_dict.values()]
lon_values = [df['lon'].values for df in test_dict.values()]

# Kontrol sonuçlar?n? yazd?rma
for i in range(len(lat_values)):
    for j in range(i + 1, len(lat_values)):
        lat_equal = (lat_values[i] == lat_values[j]).all()
        lon_equal = (lon_values[i] == lon_values[j]).all()

        if not lat_equal:
            print(f"df{i + 1} ve df{j + 1}: 'lat' columns are different.")

        if not lon_equal:
            print(f"df{i + 1} ve df{j + 1}: 'lon' columns are the same.")


lat_values = [df['lat'].values for df in train_dict.values()]
lon_values = [df['lon'].values for df in train_dict.values()]

for i in range(len(lat_values)):
    for j in range(i + 1, len(lat_values)):
        lat_equal = (lat_values[i] == lat_values[j]).all()
        lon_equal = (lon_values[i] == lon_values[j]).all()

        if not lat_equal:
            print(f"df{i + 1} ve df{j + 1}: 'lat' columns are different.")

        if not lon_equal:
            print(f"df{i + 1} ve df{j + 1}: 'lon' columns are the same.")