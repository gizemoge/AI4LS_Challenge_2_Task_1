########################################################################################################################
# EXTERNAL DATASETS
########################################################################################################################

import numpy as np
import pandas as pd
import modin.pandas as mpd
import matplotlib.pyplot as plt
import xarray as xr
import requests
import os
import pickle
import re
from tqdm import tqdm
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.6f' % x)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)


########################################################################################################################
# GLDAS
########################################################################################################################

# 1. Fetch data between 2004-2009 to match GRACE's averaged format
file_path = "Grace/datasets/subset_GLDAS_NOAH025_M_2.1_20240918_193208_.txt" # TODO belgenin adını değiştirmek
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

            ds_dict[date] = xr.open_dataset(temp_file_name)
            os.remove(temp_file_name)
            print(f'Dataset for {date} added to dictionary.')

        except requests.exceptions.RequestException as e:
            print(f'Error downloading {url}: {e}')
        except Exception as e:
            print(f'Error processing file {temp_file_name}: {e}')

print('All files processed!')

for key, value in ds_dict.items():
    ds_dict[key] = value.load()

ds = ds_dict["200401"]
variables = list(ds.data_vars)[1:]
monthly_gldas_2004_2009 = {}

for key, value in ds_dict.items():
    df = pd.DataFrame()

    for feature in variables:
        data_array = value[feature]
        feature_df = data_array.to_dataframe().reset_index().drop("time", axis=1)
        feature_df = feature_df.dropna(subset=[f"{feature}"])


        if df.empty:
            df = feature_df
        else:
            df = df.merge(feature_df, on=["lat", "lon"])

    monthly_gldas_2004_2009[key] = df.reset_index(drop=True)

with open('Grace/pkl_files/monthly_gldas_2004_2009_2209.pkl', 'wb') as f:
    pickle.dump(monthly_gldas_2004_2009, f)



# 2.1. Fetch data between 2010-2024 for data analysis
file_path = "Grace/datasets/subset_GLDAS_NOAH025_M_2.1_20240918_193208_.txt"
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

            ds_dict[date] = xr.open_dataset(temp_file_name)
            os.remove(temp_file_name)
            print(f'Dataset for {date} added to dictionary.')

        except requests.exceptions.RequestException as e:
            print(f'Error downloading {url}: {e}')
        except Exception as e:
            print(f'Error processing file {temp_file_name}: {e}')

print('All files processed!')

for key, value in ds_dict.items():
    ds_dict[key] = value.load()


# 2.2. Create dictionary of monthly dataframes containing all fetched features of interest
monthly_gldas_dict = {}
variables = list(ds_dict["201210"].data_vars)[1:]

for key, value in ds_dict.items():
    df = pd.DataFrame()

    for feature in variables:
        data_array = value[feature]
        feature_df = data_array.to_dataframe().reset_index().drop("time", axis=1)
        feature_df = feature_df.dropna(subset=[f"{feature}"])

        if df.empty:
            df = feature_df
        else:
            df = df.merge(feature_df, on=["lat", "lon"])

    monthly_gldas_dict[key] = df.reset_index(drop=True)


with open('Grace/pkl_files/gldas_dict_2010_2024.pkl', 'wb') as f:
    pickle.dump(monthly_gldas_dict, f)