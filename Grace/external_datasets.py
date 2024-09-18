########################################################################################################################
# EXTERNAL DATASETS
########################################################################################################################

import xarray as xr
import requests
from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse
import os
import pickle
import re
import numpy as np
import modin.pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)


########################################################################################################################
# 1. GLDAS 1: NOAH
########################################################################################################################

"""
The short names with extension �_tavg� are backward 3-hour averaged variables.
The short names with extension �_acc� are backward 3-hour accumulated variables.
The short names with extension �_inst� are instantaneous variables.
"""

# 1.1. Fetch data
file_path = "Grace/datasets/subset_GLDAS_NOAH025_M_2.1_20240915_100717_.txt"
ds_dict = {}
with open(file_path, "r") as f:
    # Skip the first line
    next(f)

    for line in f:
        url = line.strip()
        match = re.search(r'(\d{6})\.\d+\.nc4', url)
        if match:
            date = match.group(1)
        else:
            print(f'No date found in URL: {url}')
            continue

        print(f'Downloading {url}...')
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for request errors

            temp_file_name = f"temp_{date}.nc4"
            with open(temp_file_name, 'wb') as temp_file:
                temp_file.write(response.content)

            ds = xr.open_dataset(temp_file_name)
            ds_dict[date] = ds
            #os.remove(temp_file_name)
            print(f'Dataset for {date} added to dictionary.')

        except requests.exceptions.RequestException as e:
            print(f'Error downloading {url}: {e}')
        except Exception as e:
            print(f'Error processing file {temp_file_name}: {e}')

print('All files processed!')

# Error downloading 201210
"201210" in ds_dict.keys() # False

# Manually add missing 201210
ds = xr.open_dataset("Grace/GLDAS_NOAH025_M.A201210.021.nc4.SUB-2.nc4")
ds_dict["201210"] = ds

# Sort items by date
ds_dict = dict(sorted(ds_dict.items()))
# see the last 5 keys to check if sorted:
list(ds_dict.keys())[-5:]

# Save as pickle
with open(os.path.join('Grace', 'pkl_files', 'gldas_dict.pkl'), 'wb') as f:
    pickle.dump(ds_dict, f)


# 1.2. Process data
# 1.2.1. Process land mask to apply to all datasets
df_land["lat"].describe()
df_land_scaled = df_land[df_land["lat"] >= -59.875]

df_land["lon"].describe()
df_land_scaled["lon"].describe()
df_land_scaled["lat"].describe()

df_land_scaled = df_land_scaled.sort_values(by=['lat', 'lon'], ascending=[True, True]).reset_index(drop=True)

# 1.2.2. Create dictionary of monthly dataframes containing all fetched features of interest
monthly_gldas_dict = {}
variables = list(ds_dict["201210"].data_vars)[1:]

for key, value in ds_dict.items():
    df = pd.DataFrame()

    for feature in variables:
        data_array = value[feature]
        feature_df = data_array.to_dataframe().reset_index().drop("time", axis=1)
        # Match lat and lon with df_land
        feature_df["lat"] = feature_df["lat"].astype("float32")
        feature_df["lon"] = feature_df["lon"].apply(lambda val: val if val < 180 else val - 360).astype("float32")

        if df.empty:
            df = feature_df
        else:
            df = df.merge(feature_df, on=["lat", "lon"])

    # Filter based on land_mask
    df["land_mask"] = df_land_scaled["LO_val"].astype(int)
    df_filtered = df[df['land_mask'] == 1]
    df_filtered = df_filtered.drop("land_mask", axis=1)

    df_filtered['lat'] = df_filtered['lat'].astype('float16')
    df_filtered['lon'] = df_filtered['lon'].astype('float16')

    # Store the filtered DataFrame in swe_dict
    monthly_gldas_dict[key] = df_filtered.reset_index(drop=True)

# write pickle
with open(os.path.join('Grace', 'pkl_files', 'monthly_gldas_dict.pkl'), 'wb') as f:
    pickle.dump(monthly_gldas_dict, f)


# 1.2.3. NaN analysis
nan_means = {}
for key, df in monthly_gldas_dict.items():
    relevant_columns = df.drop(columns=["lat", "lon"])
    nan_mean = relevant_columns.isna().mean().mean() * 100
    nan_means[key] = nan_mean

for key, nan_mean in nan_means.items():
    print(f"Mean NaN percentage for {key}: {nan_mean:.3f}%") # 55.552%


# More NaNs than expected -- do these come from the source or was there a problem with previous steps?
data_array = ds_dict["200204"]["Snowf_tavg"]
filtered_row = data_array.to_dataframe().reset_index()
result = filtered_row[(filtered_row["lat"] == -55.375) & (filtered_row["lon"] == 109.875)]
print(result) # NaNs come from the source

# Are the NaNs stand-ins for 0? Are there 0s in this column/dataframe?
data_array = ds_dict["200204"]["Snowf_tavg"]
dataframe = data_array.to_dataframe().reset_index()
zero_values = (dataframe["Snowf_tavg"] == 0).sum()
zero_rows = dataframe[dataframe["Snowf_tavg"] == 0]
first_zero_row = zero_rows.iloc[0]
print(f"Number of zero values: {zero_values}") # There are 0s and NaNs together, indicating NaNs are not 0s.


# Do the rows where all features are NaN belong to the same (lat, lon) pairs across all monthly dataframes?
# Step 1: Identify common (lat, lon) pairs where all relevant columns are NaN
na_lat_lon_pairs_noah = set()

for key, df in monthly_gldas_dict.items():
    relevant_columns = df.drop(columns=["lat", "lon"])
    na_rows = df[relevant_columns.isna().all(axis=1)]
    # Store (lat, lon) pairs of these rows
    na_pairs = set(na_rows[["lat", "lon"]].itertuples(index=False, name=None))
    # Update the set of (lat, lon) pairs
    if not na_lat_lon_pairs_noah:
        na_lat_lon_pairs_noah = na_pairs
    else:
        na_lat_lon_pairs_noah = na_lat_lon_pairs_noah.intersection(na_pairs)

# Output the consistent (lat, lon) pairs
if na_lat_lon_pairs_noah:
    print("Consistent (lat, lon) pairs where all relevant columns are NaN in every DataFrame:")
    for lat, lon in na_lat_lon_pairs_noah:
        print(f"lat: {lat}, lon: {lon}")
else:
    print("No consistent (lat, lon) pairs found where all relevant columns are NaN in every DataFrame.")

len(na_lat_lon_pairs_noah) # 143722
monthly_gldas_dict["200204"].shape #(258717, 16)
143722/258717 # 0.5555181916920805

# Step 2: Drop rows where (lat, lon) match the identified pairs
for key, df in monthly_gldas_dict.items():
    # Drop rows where (lat, lon) is in na_lat_lon_pairs
    df_filtered = df[~df[["lat", "lon"]].apply(tuple, axis=1).isin(na_lat_lon_pairs_noah)].copy()

    # Convert to float16 for smaller file size
    float_cols = df_filtered.select_dtypes(include='float32').columns
    cols_to_convert = [col for col in float_cols if col != 'SWE_inst']
    df_filtered.loc[:, cols_to_convert] = df_filtered[cols_to_convert].astype('float16', errors='ignore')

    if 'index' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns='index')

    monthly_gldas_dict[key] = df_filtered


for key, df in monthly_gldas_dict.items():
    # Check for infinite values
    inf_check = df.apply(np.isinf)
    inf_counts = inf_check.sum()
    inf_counts_nonzero = inf_counts[inf_counts > 0]

    if not inf_counts_nonzero.empty:
        print(f"DataFrame for {key}:")
        for col, count in inf_counts_nonzero.items():
            print(f"  {col}: {count} infinite values\n")
    else:
        print(f"No infinite values found in DataFrame for {key}.\n")


with open(os.path.join('Grace', 'pkl_files', 'monthly_gldas_dict_filtered_float16.pkl'), 'wb') as f:
    pickle.dump(monthly_gldas_dict, f)

with open("Grace/pkl_files/monthly_gldas_dict_filtered_float16.pkl", "rb") as f:
    gldas_dict = pickle.load(f)


########################################################################################################################
# 2. ESGF
########################################################################################################################

ds2014 = xr.open_dataset('Grace/datasets/tas_Amon_GISS-E2-1-G_historical_r1i1p5f1_gn_200101-201412.nc')

ds2014.variables
ds2014.info()
ds2014.data_vars
# S?cakl?k verilerine eri?in

tas_df_2014 = ds2014['tas'].to_dataframe().reset_index()
tas_df_2014.isnull().sum()

tas_df_2014.head()
tas_df_2014["time"].value_counts()
tas_df_2014[tas_df_2014["lat"] < -88]

tas_df_2014.shape


tas_df_2014_changes = tas_df_2014[tas_df_2014['lat'] != tas_df_2014['lat'].shift()]
tas_df_2014_changes.head()


##############################################################

ds = xr.open_dataset('Grace/datasets/tas_Amon_IPSL-CM6A-LR_dcppA-hindcast_s2014-r6i1p1f1_gr_201501-202412.nc')

ds.variables
ds.data_vars
ds.info()
df_2024 = ds["tas"].to_dataframe().reset_index()
df_2024.head(20)
df_2024.tail()
df_2024.shape # (2471040, 5)


########################################################################################################################
# 3. POPULATION
########################################################################################################################

ds_population = xr.open_dataset('Grace/datasets/gpw_v4_population_count_adjusted_rev11_30_min.nc')

ds_population.data_vars
df_population = ds_population["UN WPP-Adjusted Population Count, v4.11 (2000, 2005, 2010, 2015, 2020): 30 arc-minutes"].to_dataframe().reset_index()

df_population.isnull().sum()
df_population.shape

df_population["raster"].value_counts()

df_population.head()

############################################

with open("Grace/pkl_files/monthly_gldas_dict_filtered_float16.pkl", "rb") as f:
    gldas_dict = pickle.load(f)

gldas_dict.keys()
gldas_dict["200204"].head()

########################################################################################################################
# AIR TEMPARATURE
########################################################################################################################

data = xr.open_dataset("Grace/datasets/air.mon.mean.nc")

data.info()
data.data_vars

df_air = data["air"].to_dataframe().reset_index()
df_air.head()
df_air.isnull().sum()
df_air.shape

df_air = df_air[df_air['time'] >= '2002-04-01'].reset_index()
df_air.drop('index', inplace=True, axis=1)
df_air.shape


df_air['lat'] = df_air['lat'].astype('float16')
df_air['lon'] = df_air['lon'].astype('float16')
df_air['air'] = df_air['air'].astype('float16')



df_air.shape
df_air.info()
df_air.head()
df_air.isnull().sum()
# 'air' sütunu NaN olan satırları sil ve indeksleri sıfırla
df_air = df_air.dropna(subset=['air']).reset_index(drop=True)
df_air.to_pickle('Grace/pkl_files/df_air.pkl')


df_air = pd.read_pickle("Grace/pkl_files/df_air.pkl")

df_air.info()
df_air.head(30)
df_air.tail()

df_air.describe()
df_air.isnull().sum()

########################################################################################################################
# 3. GLDAS 2: CATCHMENT
########################################################################################################################


file_path = "Grace/datasets/subset_GLDAS_NOAH025_M_2.1_20240917_200833_.txt"

ds_dict = {}
with open(file_path, "r") as f:
    # Skip the first line
    next(f)

    for line in f:
        url = line.strip()
        match = re.search(r'(\d{6})\.\d+\.nc4', url)
        if match:
            date = match.group(1)
        else:
            print(f'No date found in URL: {url}')
            continue

        print(f'Downloading {url}...')
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for request errors

            temp_file_name = f"temp_{date}.nc4"
            with open(temp_file_name, 'wb') as temp_file:
                temp_file.write(response.content)

            ds = xr.open_dataset(temp_file_name)
            ds_dict[date] = ds
            #os.remove(temp_file_name)
            print(f'Dataset for {date} added to dictionary.')

        except requests.exceptions.RequestException as e:
            print(f'Error downloading {url}: {e}')
        except Exception as e:
            print(f'Error processing file {temp_file_name}: {e}')

print('All files processed!')

with open(os.path.join('Grace', 'pkl_files', 'gldas_catchment_dict.pkl'), 'wb') as f:
    pickle.dump(ds_dict, f)


df_land_scaled = df_land[df_land["lat"] >= -59.875]
df_land_scaled = df_land_scaled.sort_values(by=['lat', 'lon'], ascending=[True, True]).reset_index(drop=True)

monthly_gldas_catchment_dict = {}
variables = list(ds_dict["201210"].data_vars)[1:]

for key, value in ds_dict.items():
    df = pd.DataFrame()

    for feature in variables:
        data_array = value[feature]
        feature_df = data_array.to_dataframe().reset_index().drop("time", axis=1)
        # Match lat and lon with df_land
        feature_df["lat"] = feature_df["lat"].astype("float32")
        feature_df["lon"] = feature_df["lon"].apply(lambda val: val if val < 180 else val - 360).astype("float32")

        if df.empty:
            df = feature_df
        else:
            df = df.merge(feature_df, on=["lat", "lon"])

    # Filter based on land_mask
    df["land_mask"] = df_land_scaled["LO_val"].astype(int)
    df_filtered = df[df['land_mask'] == 1]
    df_filtered = df_filtered.drop("land_mask", axis=1)

    # Store the filtered DataFrame in swe_dict
    monthly_gldas_catchment_dict[key] = df_filtered.reset_index(drop=True)

# write pickle
with open(os.path.join('Grace', 'pkl_files', 'monthly_gldas_catchment_dict_processed.pkl'), 'wb') as f:
    pickle.dump(monthly_gldas_catchment_dict, f)


na_lat_lon_pairs_catchment = set()

for key, df in monthly_gldas_catchment_dict.items():
    relevant_columns = df.drop(columns=["lat", "lon"])
    na_rows = df[relevant_columns.isna().all(axis=1)]
    # Store (lat, lon) pairs of these rows
    na_pairs = set(na_rows[["lat", "lon"]].itertuples(index=False, name=None))
    # Update the set of (lat, lon) pairs
    if not na_lat_lon_pairs_catchment:
        na_lat_lon_pairs_catchment = na_pairs
    else:
        na_lat_lon_pairs_catchment = na_lat_lon_pairs_catchment.intersection(na_pairs)

# Output the consistent (lat, lon) pairs
if na_lat_lon_pairs_catchment:
    print("Consistent (lat, lon) pairs where all relevant columns are NaN in every DataFrame:")
    for lat, lon in na_lat_lon_pairs_catchment:
        print(f"lat: {lat}, lon: {lon}")
else:
    print("No consistent (lat, lon) pairs found where all relevant columns are NaN in every DataFrame.")

print(len(na_lat_lon_pairs_catchment)) # 141972
print(monthly_gldas_catchment_dict["200204"].shape) # (258717, 7)

na_lat_lon_pairs_noah == na_lat_lon_pairs_catchment # True

# Step 2: Drop rows where (lat, lon) match the identified pairs
for key, df in monthly_gldas_catchment_dict.items():
    # Drop rows where (lat, lon) is in na_lat_lon_pairs
    df_filtered = df[~df[["lat", "lon"]].apply(tuple, axis=1).isin(na_lat_lon_pairs_catchment)].copy()

    if 'index' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns='index')

    monthly_gldas_catchment_dict[key] = df_filtered

# Check for infinite values as a result of float16 conversion
for key, df in monthly_gldas_catchment_dict.items():
    inf_check = df.apply(np.isinf)
    inf_counts = inf_check.sum()
    inf_counts_nonzero = inf_counts[inf_counts > 0]

    if not inf_counts_nonzero.empty:
        print(f"DataFrame for {key}:")
        for col, count in inf_counts_nonzero.items():
            print(f"  {col}: {count} infinite values\n")
    else:
        print(f"No infinite values found in DataFrame for {key}.\n")


with open('Grace/pkl_files/monthly_gldas_catchment_dict_filtered_float16.pkl', 'wb') as f:
    pickle.dump(monthly_gldas_catchment_dict, f)



with open("Grace/pkl_files/monthly_gldas_noah_dict_filtered_float16.pkl", "rb") as f:
    monthly_gldas_noah_dict = pickle.load(f)

with open("Grace/pkl_files/monthly_gldas_catchment_dict_filtered_float16.pkl", "rb") as f:
    monthly_gldas_catchment_dict = pickle.load(f)

monthly_gldas_noah_dict.keys()
monthly_gldas_catchment_dict.keys()

monthly_gldas_noah_dict["200204"].head()
monthly_gldas_catchment_dict["200204"].head()


# COMBINE THE TWO GLDAS DICTS
combined_gldas_monthly = {}
common_keys = set(monthly_gldas_noah_dict.keys()).intersection(set(monthly_gldas_catchment_dict.keys()))

# Float16 doesnt work for merge
for key, df in monthly_gldas_noah_dict.items():
    monthly_gldas_noah_dict[key] = pd.DataFrame(df).astype('float32')

for key, df in monthly_gldas_catchment_dict.items():
    monthly_gldas_catchment_dict[key] = pd.DataFrame(df).astype('float32')

for key in common_keys:
    merged_df = pd.merge(monthly_gldas_noah_dict[key], monthly_gldas_catchment_dict[key], on=['lat', 'lon'], how='inner')
    # Convert to float16 for smaller file size
    float_cols = merged_df.select_dtypes(include='float32').columns
    cols_to_convert = [col for col in float_cols if col != 'SWE_inst']
    merged_df.loc[:, cols_to_convert] = merged_df[cols_to_convert].astype('float16', errors='ignore')
    combined_gldas_monthly[key] = merged_df

combined_gldas_monthly.keys()
combined_gldas_monthly = dict(sorted(combined_gldas_monthly.items()))
combined_gldas_monthly["200204"].head()

with open('Grace/pkl_files/combined_gldas_monthly_filtered_float16.pkl', 'wb') as f:
    pickle.dump(combined_gldas_monthly, f)