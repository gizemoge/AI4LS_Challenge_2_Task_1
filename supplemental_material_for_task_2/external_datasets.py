########################################################################################################################
# EXTERNAL DATASETS
########################################################################################################################

import pandas as pd
import xarray as xr
import requests
import os
import re

def load_gldas_dict_2004_2009():
    # 1. Fetch data between 2004-2009 to match GRACE's averaged format
    file_path = "supplemental_material_for_task_2/datasets/2004_2009_avg_gldas_noah_2209.txt"
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
    return monthly_gldas_2004_2009


def load_gldas_dict_2010_2024():
    # 2. Fetch data between 2010-2024 for data analysis
    file_path = "supplemental_material_for_task_2/datasets/subset_GLDAS_NOAH025_M_2.1_20240918_193208_.txt"
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

    ds = ds_dict["201210"]
    variables = list(ds.data_vars)[1:]
    gldas_dict_2010_2024 = {}

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

        gldas_dict_2010_2024[key] = df.reset_index(drop=True)

    return gldas_dict_2010_2024