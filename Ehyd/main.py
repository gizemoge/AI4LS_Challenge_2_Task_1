# -*- coding: iso-8859-1 -*-

# Importing Libraries and Arranging Console Display.
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.spatial import distance
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)

# FUNCTIONS
def station_coordinates(input):
    """
    Creates a dataset consisting of measurement station IDs and their corresponding X and Y coordinates.

    Args:
        input: Directory of the measurement station CSV file.

    Returns:
        df: A DataFrame containing columns "x", "y", and "hzbnr01".
    """
    df = pd.read_csv(f"datasets_ehyd/{input}/messstellen_alle.csv", sep=";")
    output_df = df[["x", "y", "hzbnr01"]].copy()
    output_df['x'] = output_df['x'].astype(str).str.replace(',', '.').astype("float32")
    output_df['y'] = output_df['y'].astype(str).str.replace(',', '.').astype("float32")
    return output_df

def to_dataframe(folder_path, tip_coordinates):
    """
    Processes CSV files in the specified folder, skipping header information and creating DataFrames
    from the section marked by "Werte". Converts "Lücke" (Gap) values to NaN and skips rows with
    invalid data or specific keywords.

    For each CSV file, it extracts data starting after the "Werte:" line, processes date and value
    columns, and stores each DataFrame in a dictionary where the key is derived from the filename.
    Additionally, it matches IDs with tip coordinates and returns a DataFrame containing matched coordinates.

    Args:
        folder_path (str): The directory path where the CSV files are located.
        tip_coordinates (pd.DataFrame): A DataFrame containing coordinates to be matched with the IDs.

    Returns:
        dict: A dictionary where keys are IDs (extracted from filenames) and values are DataFrames.
        pd.DataFrame: A DataFrame with matched coordinates based on IDs.
    """
    dataframes_dict = {}
    coordinates = pd.DataFrame()

    for filename in os.listdir(folder_path):
        try:
            if filename.endswith(".csv"):
                filepath = os.path.join(folder_path, filename)

                with open(filepath, 'r', encoding='latin1') as file:
                    lines = file.readlines()

                    # Find the starting index of the data section
                    start_idx = next((i for i, line in enumerate(lines) if line.startswith("Werte:")), None)
                    if start_idx is None:
                        continue  # Skip files that do not contain 'Werte:'

                    start_idx += 1
                    header_line = lines[start_idx - 1].strip()

                    # Skip files with 'Invalid' in the header line
                    if "Invalid" in header_line:
                        continue

                    data_lines = lines[start_idx:]

                    data = []
                    for line in data_lines:
                        if line.strip():  # Skip empty lines
                            try:
                                date_str, value_str = line.split(';')[:2]

                                # Try multiple date formats
                                try:
                                    date = datetime.strptime(date_str.strip(), "%d.%m.%Y %H:%M:%S").date()
                                except ValueError:
                                    try:
                                        date = datetime.strptime(date_str.strip(), "%d.%m.%Y %H:%M").date()
                                    except ValueError:
                                        continue

                                # Skip rows with invalid data or specific keywords
                                if any(keyword in value_str for keyword in ["F", "K", "rekonstruiert aus Version 3->"]):
                                    continue

                                # Convert value to float
                                try:
                                    value = np.float32(value_str.replace(',', '.'))
                                except ValueError:
                                    value = np.nan  # Assign NaN if conversion fails

                                data.append([date, value])

                            except Exception:
                                break

                    if data:  # Create DataFrame only if data exists
                        df = pd.DataFrame(data, columns=['Date', 'Values'])
                        df.drop(df.index[-1], inplace=True)  # Dropping the last row (2022-01-01)
                        df_name = f"{filename[-10:-4]}"

                        dataframes_dict[df_name] = df

                        # Convert keys to integers
                        int_keys = [int(key) for key in dataframes_dict.keys() if key.isdigit()]
                        coordinates = tip_coordinates[tip_coordinates['hzbnr01'].isin(int_keys)]

        except Exception:
            continue

    return dataframes_dict, coordinates

def to_global(dataframes_dict, prefix=''):
    """
    Adds DataFrames from a dictionary to the global namespace with optional prefix.

    Args:
        dataframes_dict (dict): A dictionary where keys are names (str) and values are DataFrames.
        prefix (str): An optional string to prefix to each DataFrame name in the global namespace.
    """
    for name, dataframe in dataframes_dict.items():
        globals()[f"{prefix}{name}"] = dataframe

def process_dataframes(df_dict):
    """
    Processes a dictionary of DataFrames by converting date columns, resampling daily data to monthly, and reindexing.

    Args:
        df_dict (dict): A dictionary where keys are DataFrame names and values are DataFrames.

    Returns:
        dict: The processed dictionary of DataFrames with date conversion, resampling, and reindexing applied.
    """
    for df_name, df_value in df_dict.items():
        df_value['Date'] = pd.to_datetime(df_value['Date'])

        if df_value['Date'].dt.to_period('D').nunique() > df_value['Date'].dt.to_period('M').nunique():
            df_value.set_index('Date', inplace=True)
            df_dict[df_name] = df_value.resample('MS').mean()

        else:
            df_value.set_index('Date', inplace=True)
            df_dict[df_name] = df_value

        all_dates = pd.date_range(start='1960-01-01', end='2021-12-01', freq='MS')
        new_df = pd.DataFrame(index=all_dates)
        df_dict[df_name] = new_df.join(df_dict[df_name], how='left').fillna("NaN")

    return df_dict

def process_and_store_data(folder, coordinates, prefix, station_list=None):
    data_dict, data_coordinates = to_dataframe(folder, coordinates)
    data_dict = process_dataframes(data_dict)

    for df_name, df in data_dict.items():
        df.astype('float32')

    to_global(data_dict, prefix=prefix)

    if station_list:
        data_dict = filter_dataframes_by_stations(data_dict, station_list)
        data_coordinates = data_coordinates[data_coordinates['hzbnr01'].astype(str).isin(station_list)]

    return data_dict, data_coordinates

def filter_dataframes_by_stations(dataframes_dict, station_list):
    """
    Filters a dictionary of DataFrames to include only those whose names are specified in a given CSV file.

    Args:
        dataframes_dict (dict): A dictionary where keys are names (str) and values are DataFrames.
        station_list (str): Path to a CSV file that contains the names (str) of the DataFrames to filter.

    Returns:
        dict: A filtered dictionary containing only the DataFrames whose names are listed in the CSV file.
    """
    filtered_dict = {name: df for name, df in dataframes_dict.items() if name in station_list}
    return filtered_dict

def main():
    ########################################################################################################################
    # Creating Dataframes from given CSVs
    ########################################################################################################################
    # Define paths and coordinates
    groundwater_all_coordinates = station_coordinates("Groundwater")
    precipitation_coordinates = station_coordinates("Precipitation")
    sources_coordinates = station_coordinates("Sources")
    surface_water_coordinates = station_coordinates("Surface_Water")

    precipitation_folders = [
        ("N-Tagessummen", "rain_"),
        ("NS-Tagessummen", "snow_")]

    source_folders = [
        ("Quellschüttung-Tagesmittel", "source_fr_"),
        ("Quellleitfähigkeit-Tagesmittel", "conductivity_"),
        ("Quellwassertemperatur-Tagesmittel", "source_temp_")]

    surface_water_folders = [
        ("W-Tagesmittel", "surface_water_level_"),
        ("WT-Monatsmittel", "surface_water_temp_"),
        ("Schwebstoff-Tagesfracht", "sediment_"),
        ("Q-Tagesmittel", "surface_water_fr_")]

    # Groundwater Dictionary (Filtered down to the requested 487 stations)
    stations = pd.read_csv("datasets_ehyd/gw_test_empty.csv")
    station_list = [col for col in stations.columns[1:]]
    filtered_groundwater_dict, filtered_gw_coordinates = process_and_store_data(
        "datasets_ehyd/Groundwater/Grundwasserstand-Monatsmittel",
        groundwater_all_coordinates, "gw_", station_list)

    gw_temp_dict, gw_temp_coordinates = process_and_store_data(os.path.join("datasets_ehyd", "Groundwater", "Grundwassertemperatur-Monatsmittel"), groundwater_all_coordinates, "gwt_")
    rain_dict, rain_coord = process_and_store_data(os.path.join("datasets_ehyd", "Precipitation", precipitation_folders[0][0]), precipitation_coordinates, "rain_")
    snow_dict, snow_coord = process_and_store_data(os.path.join("datasets_ehyd", "Precipitation", precipitation_folders[1][0]), precipitation_coordinates, "snow_")
    source_fr_dict, source_fr_coord = process_and_store_data(os.path.join("datasets_ehyd", "Sources", source_folders[0][0]), sources_coordinates, "source_fr_")
    conduct_dict, conduct_coord = process_and_store_data(os.path.join("datasets_ehyd", "Sources", source_folders[1][0]), sources_coordinates, "conduct_")
    source_temp_dict, source_temp_coord = process_and_store_data(os.path.join("datasets_ehyd", "Sources", source_folders[2][0]), sources_coordinates, "source_temp_")
    surface_water_lvl_dict, surface_water_lvl_coord = process_and_store_data(os.path.join("datasets_ehyd", "Surface_Water", surface_water_folders[0][0]), surface_water_coordinates, "surface_water_lvl_")
    surface_water_temp_dict, surface_water_temp_coord = process_and_store_data(os.path.join("datasets_ehyd", "Surface_Water", surface_water_folders[1][0]), surface_water_coordinates, "surface_water_temp_")
    sediment_dict, sediment_coord = process_and_store_data(os.path.join("datasets_ehyd", "Surface_Water", surface_water_folders[2][0]), surface_water_coordinates, "sediment_")
    surface_water_fr_dict, surface_water_fr_coord = process_and_store_data(os.path.join("datasets_ehyd", "Surface_Water", surface_water_folders[3][0]), surface_water_coordinates, "surface_water_fr_")

    print("--------------------- DataFrame process and store: Complete")
    ########################################################################################################################
    # Gathering associated additional features for required 487 stations
    ########################################################################################################################
    def calculate_distance(coord1, coord2):
        """
        Calculates the Euclidean distance between two points in a Cartesian coordinate system.

        Args:
            coord1 (tuple): A tuple representing the coordinates (x, y) of the first point.
            coord2 (tuple): A tuple representing the coordinates (x, y) of the second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return distance.euclidean(coord1, coord2)

    def find_nearest_coordinates(gw_row, df, k=20):
        """
        Finds the `k` nearest coordinates from a DataFrame to a given point.

        Args:
            gw_row (pd.Series): A pandas Series representing the coordinates (x, y) of the given point.
            df (pd.DataFrame): A DataFrame containing the coordinates with columns "x" and "y".
            k (int, optional): The number of nearest coordinates to return. Defaults to 20.

        Returns:
            pd.DataFrame: A DataFrame containing the `k` nearest coordinates to the given point.
        """
        distances = df.apply(lambda row: calculate_distance(
            (gw_row['x'], gw_row['y']),
            (row['x'], row['y'])
        ), axis=1)
        nearest_indices = distances.nsmallest(k).index
        return df.loc[nearest_indices]

    # Creating a dataframe that stores all the associated features' information of the 487 stations.
    data = pd.DataFrame()
    def add_nearest_coordinates_column(df_to_add, name, k, df_to_merge=None):
        if df_to_merge is None:
            df_to_merge = data  # Use the current value of 'data' as the default
        results = []

        # Find the nearest stations according to the coordinates
        for _, gw_row in filtered_gw_coordinates.iterrows():
            nearest = find_nearest_coordinates(gw_row, df_to_add, k)
            nearest_list = nearest['hzbnr01'].tolist()
            results.append({
                'hzbnr01': gw_row['hzbnr01'],
                name: nearest_list
            })

        results_df = pd.DataFrame(results)

        # # Debug: Check if 'hzbnr01' exists in both dataframes
        # print("Columns in df_to_merge:", df_to_merge.columns)
        # print("Columns in results_df:", results_df.columns)

        # Ensure that the column exists in both dataframes before merging
        if 'hzbnr01' in df_to_merge.columns and 'hzbnr01' in results_df.columns:
            # Merge operation
            df = df_to_merge.merge(results_df, on='hzbnr01', how='inner')

            # Debug: Birle?tirilmi? DataFrame'i yazd?rarak kontrol et
            # print("Merged DataFrame:")
            # print(df.head())
        else:
            raise KeyError("Column 'hzbnr01' does not exist in one of the dataframes.")

        return df

    data = add_nearest_coordinates_column(gw_temp_coordinates, 'nearest_gw_temp', 1, df_to_merge=filtered_gw_coordinates)
    data = add_nearest_coordinates_column(rain_coord, 'nearest_rain', 3)
    data = add_nearest_coordinates_column(snow_coord, 'nearest_snow', 3)
    data = add_nearest_coordinates_column(source_fr_coord, 'nearest_source_fr', 1)
    data = add_nearest_coordinates_column(conduct_coord, 'nearest_conductivity', 1)
    data = add_nearest_coordinates_column(source_temp_coord, 'nearest_source_temp', 1)
    data = add_nearest_coordinates_column(surface_water_lvl_coord, 'nearest_owf_level', 3)
    data = add_nearest_coordinates_column(surface_water_temp_coord, 'nearest_owf_temp', 1)
    data = add_nearest_coordinates_column(sediment_coord, 'nearest_sediment', 1)
    data = add_nearest_coordinates_column(surface_water_fr_coord, 'nearest_owf_fr', 3,)
    data.drop(["x", "y"], axis=1, inplace=True)

    ########################################################################################################################
    # Imputing NaN Values
    ########################################################################################################################
    def nan_imputer(dict):
        """
        Imputes missing values in a dictionary of DataFrames by filling NaNs with the corresponding monthly means.

        Args:
            dict (dict): A dictionary where the keys are DataFrame names and the values are DataFrames
                         containing a 'Values' column with missing values represented as 'NaN'.

        Returns:
            dict: A dictionary with the same keys as the input, but with NaN values in each DataFrame
                  replaced by the monthly mean of the 'Values' column.
        """
        new_dict = {}
        for df_name, df in dict.items():
            df_copy = df.copy(deep=True)  # Create a deep copy
            df_copy.replace('NaN', np.nan, inplace=True)
            first_valid_index = df_copy['Values'].first_valid_index()
            valid_values = df_copy.loc[first_valid_index:].copy()

            # Fill NaNs with the corresponding monthly means
            for month in range(1, 13):
                month_mean = valid_values[valid_values.index.month == month]['Values'].dropna().mean()
                valid_values.loc[valid_values.index.month == month, 'Values'] = valid_values.loc[
                    valid_values.index.month == month, 'Values'].fillna(month_mean)

            # Update the copied DataFrame with filled values
            df_copy.update(valid_values)
            new_dict[df_name] = df_copy  # Store the modified copy

        return new_dict

    filled_filtered_groundwater_dict = nan_imputer(filtered_groundwater_dict)
    filled_gw_temp_dict = nan_imputer(gw_temp_dict)
    filled_rain_dict = nan_imputer(rain_dict)
    filled_snow_dict = nan_imputer(snow_dict)
    filled_source_fr_dict = nan_imputer(source_fr_dict)
    filled_source_temp_dict = nan_imputer(source_temp_dict)
    filled_conduct_dict = nan_imputer(conduct_dict)
    filled_surface_water_fr_dict = nan_imputer(surface_water_fr_dict)
    filled_surface_water_lvl_dict = nan_imputer(surface_water_lvl_dict)
    filled_surface_water_temp_dict = nan_imputer(surface_water_temp_dict)
    filled_sediment_dict = nan_imputer(sediment_dict)

    print("--------------------- NaN imputer: Complete")
    ########################################################################################################################
    # Adding lagged values and rolling means
    ########################################################################################################################
    filled_dict_list = [filled_gw_temp_dict, filled_filtered_groundwater_dict, filled_snow_dict, filled_rain_dict,
                        filled_conduct_dict, filled_source_fr_dict, filled_source_temp_dict, filled_surface_water_lvl_dict,
                        filled_surface_water_fr_dict, filled_surface_water_temp_dict, filled_sediment_dict]

    def add_lag_and_rolling_mean(df, window=6):
        """
        Adds lag features and rolling mean to a DataFrame.

        Args:
            df (pd.DataFrame): A DataFrame with at least one column, which will be used to create lag features
                               and compute rolling mean. The first column of the DataFrame will be used.
            window (int, optional): The window size for computing the rolling mean. Defaults to 6.

        Returns:
            pd.DataFrame: The original DataFrame with additional columns for lag features and rolling mean.
                          Includes lag features for 1, 2, and 3 periods and rolling mean columns with the specified window size.
        """
        column_name = df.columns[0]
        df['lag_1'] = df[column_name].shift(1)
        df['lag_2'] = df[column_name].shift(2)
        df['lag_3'] = df[column_name].shift(3)

        df["rolling_mean_original"] = df[column_name].rolling(window=window).mean()

        for i in range(1, 4):
            df[f'rolling_mean_{window}_lag_{i}'] = df["rolling_mean_original"].shift(i)
        return df

    for dictionary in filled_dict_list:
        for key, df in dictionary.items():
            dictionary[key] = add_lag_and_rolling_mean(df)

    ########################################################################################################################
    # Zero Padding and Data Type Change (float32)
    ########################################################################################################################
    for dictionary in filled_dict_list:
        for key, df in dictionary.items():
            df.fillna(0, inplace=True)
            df = df.astype(np.float32)
            dictionary[key] = df

    ########################################################################################################################
    # Creating two new dictionaries:
    #   new_dataframes: is a dictionary storing DataFrames specific to each measurement station, containing both the
    #                   station's data and associated features obtained from the data DataFrame.
    #   monthly_dataframes: contains monthly versions of the data from new_dataframes, with keys representing months
    #                   instead of measurement station IDs.
    ########################################################################################################################
    data['hzbnr01'] = data['hzbnr01'].apply(lambda x: [x])

    data_sources = {
        'nearest_gw_temp': ('gw_temp', filled_gw_temp_dict),
        'nearest_rain': ('rain', filled_rain_dict),
        'nearest_snow': ('snow', filled_snow_dict),
        'nearest_conductivity': ('conduct', filled_conduct_dict),
        'nearest_source_fr': ('source_fr', filled_source_fr_dict),
        'nearest_source_temp': ('source_temp', filled_source_temp_dict),
        'nearest_owf_level': ('owf_level', filled_surface_water_lvl_dict),
        'nearest_owf_temp': ('owf_temp', filled_surface_water_temp_dict),
        'nearest_owf_fr': ('owf_fr', filled_surface_water_fr_dict),
        'nearest_sediment': ('sediment', filled_sediment_dict)
    }

    new_dataframes = {}
    for idx, row in data.iterrows():
        code = str(row['hzbnr01'][0])

        if code in filled_filtered_groundwater_dict:
            df = filled_filtered_groundwater_dict[code].copy()

            for key, (prefix, source_dict) in data_sources.items():
                for i, code_value in enumerate(row[key]):
                    code_str = str(code_value)
                    source_df = source_dict.get(code_str, pd.DataFrame())

                    source_df = source_df.rename(columns=lambda x: f'{prefix}_{i + 1}_{x}')
                    df = df.join(source_df, how='left')

                    columns = ["Values", "lag_1", "lag_2", "lag_3", "rolling_mean_original", "rolling_mean_6_lag_1", "rolling_mean_6_lag_2", "rolling_mean_6_lag_3"]
                    for column in columns:
                        if i == 2:
                            df[f"{prefix}_{column}_mean"] = (df[f"{prefix}_{i + 1}_{column}"] + df[f"{prefix}_{i}_{column}"] + df[f"{prefix}_{i - 1}_{column}"]) / 3

            new_dataframes[code] = df


    print("--------------------- new_dataframes dictionary: Complete")
    monthly_dict_85to21 = {}
    for year in range(1985, 2022):
        for month in range(1, 13):

            key = f"{year}_{month:02d}"
            monthly_data = []

            for df_id, df in new_dataframes.items():
                mask = (df.index.to_period("M").year == year) & (df.index.to_period("M").month == month)

                if mask.any():
                    filtered_df = df[mask]
                    new_index = [f"{df_id}" for i in range(len(filtered_df))]
                    filtered_df.index = new_index
                    monthly_data.append(filtered_df)

            if monthly_data:
                combined_df = pd.concat(monthly_data)
                monthly_dict_85to21[key] = combined_df

    print("--------------------- monthly_dict_85to21 dictionary: Complete")
    print("--------------------- Starting SARIMA!")
    ########################################################################################################################
    # SARIMA Model
    ########################################################################################################################
    def average_correlation_feature_selection(data_dict, threshold=0.1):
        """
        Computes average correlation of features with the target variable across multiple dataframes and selects features
        based on a correlation threshold.

        Args:
            data_dict (dict): A dictionary where each value is a pandas DataFrame. Each DataFrame should contain a column
                              named 'Values' representing the target variable.
            threshold (float, optional): The minimum absolute correlation value required for a feature to be selected.
                                          Defaults to 0.1.

        Returns:
            list: A list of feature names that have an average correlation with the target variable above the specified
                  threshold.
        """
        feature_corr_sum = None
        feature_count = 0

        for df in data_dict.values():
            corr_matrix = df.corr()
            target_corr = corr_matrix['Values'].drop('Values')

            if feature_corr_sum is None:
                feature_corr_sum = target_corr
            else:
                feature_corr_sum += target_corr

            feature_count += 1

        avg_corr = feature_corr_sum / feature_count

        selected_features = avg_corr[avg_corr.abs() > threshold].index.tolist()

        return selected_features

    common_features = average_correlation_feature_selection(monthly_dict_85to21, threshold=0.4)  # 39
    """
    new_start_date = pd.to_datetime('1985-01-01')
    
    adjusted_dataframes = {}
    
    for key, df in new_dataframes.items():
        try:
            df.index = pd.to_datetime(df.index)
            df_filtered = df[df.index >= new_start_date]
            adjusted_dataframes[key] = df_filtered
        except Exception as e:
            print(f"An error occurred with key {key}: {e}")
    
    
    filtered_dataframes = {}
    
    def filter_dataframe_by_features(df, features):
        target_column = df.columns[0]
        filtered_features = [target_column] + [col for col in features if col != target_column]
        df_filtered = df[filtered_features]
    
        return df_filtered
    
    for key, df in adjusted_dataframes.items():
        try:
            filtered_df = filter_dataframe_by_features(df, common_features)
            filtered_dataframes[key] = filtered_df
        except KeyError as e:
            print(f"KeyError: {e} - Some columns in {key} are missing.")
        except Exception as e:
            print(f"An error occurred with key {key}: {e}")
    
    for key, value in filtered_dataframes.items():
        print(value.head())
    
    # Hyperparameter Optimization
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    def sarima_optimizer_aic(train, exog, pdq, seasonal_pdq):
        best_aic, best_order, best_seasonal_order = float("inf"), None, None
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    sarimax_model = SARIMAX(train, exog=exog, order=param, seasonal_order=param_seasonal)
                    results = sarimax_model.fit(disp=0)
                    aic = results.aic
                    if aic < best_aic:
                        best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                    print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
                except Exception as e:
                    print(f"Exception: {e} - SARIMA{param}x{param_seasonal}12")
                    continue
        print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
        return best_order, best_seasonal_order
    
    results = {}
    
    sampled_keys = random.sample(list(filtered_dataframes.keys()), 10)
    
    for df_id in sampled_keys:
        df = filtered_dataframes[df_id]
        if not df.empty:
            train = df['Values']
            exog = df.drop(columns=['Values'])
            best_order, best_seasonal_order = sarima_optimizer_aic(train, exog, pdq, seasonal_pdq)
            results[df_id] = {
                'Best Order': best_order,
                'Best Seasonal Order': best_seasonal_order
            }
        else:
            results[df_id] = {
                'Best Order': None,
                'Best Seasonal Order': None
            }
    
    for df_id, res in results.items():
        print(f'DataFrame ID: {df_id}')
        print(f'Best Order: {res["Best Order"]}')
        print(f'Best Seasonal Order: {res["Best Seasonal Order"]}')
        print('---')
    
    # order=(1, 0, 1) and seasonal_order=(0, 1, 1, 12) are the top picks.
    
    
    for month, df in monthly_dict_with_correlation.items():
        monthly_dict_with_correlation[month] = df[ ['Values'] + common_features]
    
    # Test months to forecast
    forecast_months = ['2020_01', '2020_02', '2020_03', '2020_04', '2020_05', '2020_06',
                       '2020_07', '2020_08', '2020_09', '2020_10', '2020_11', '2020_12',
                       '2021_01', '2021_02', '2021_03', '2021_04', '2021_05', '2021_06',
                       '2021_07', '2021_08', '2021_09', '2021_10', '2021_11', '2021_12']
    
    train_data = {month: df for month, df in monthly_dict_with_correlation.items() if month not in forecast_months}
    
    train_data = {k: v for k, v in train_data.items() if k >= "2015_01"}
    
    all_data = pd.concat([df for df in train_data.values()])
    
    forecasts = {}
    for station in all_data.index.unique():
        station_data = all_data.loc[station]
    
        model = SARIMAX(
            station_data['Values'],
            exog=station_data.drop(columns=['Values']),
            order=(1, 0, 1),
            seasonal_order=(0, 1, 1, 12)
        )
        model_fit = model.fit(disp=False)
    
        forecast = model_fit.get_forecast(steps=24, exog=station_data.drop(columns=['Values']).values[
                                                         -24:])
        forecast_values = forecast.predicted_mean
        forecasts[station] = forecast_values
    
    forecast_df = pd.DataFrame(forecasts).T
    forecast_df.columns = [f'forecast_month_{i + 1}' for i in range(24)]
    
    print(forecast_df)
    
    test_data = {month: df for month, df in monthly_dict_with_correlation.items() if month in forecast_months}
    test_data = pd.concat([df for df in test_data.values()])
    
    # Calculating SMAPE
    def smape(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    all_actual = []
    all_predicted = []
    
    for station in forecast_df.index:
        actual_values = test_data.loc[station, 'Values'].values
        predicted_values = forecast_df.loc[station].values
    
        all_actual.extend(actual_values)
        all_predicted.extend(predicted_values)
    
    general_smape = smape(np.array(all_actual), np.array(all_predicted))
    print(f"Overall SMAPE: {general_smape:.2f}%")
    # Overall SMAPE: 0.15%
    """

    ########################################################################################################################
    # Forecasting
    ########################################################################################################################
    for month, df in monthly_dict_85to21.items():
        monthly_dict_85to21[month] = df[ ['Values'] + common_features]

    monthly_dict_final_train = {k: v for k, v in monthly_dict_85to21.items() if k >= "2015_01"}

    final_data = pd.concat([df for df in monthly_dict_final_train.values()])

    forecasts_final = {}
    for station in final_data.index.unique():
        station_data = final_data.loc[station]

        model = SARIMAX(
            station_data['Values'],
            exog=station_data.drop(columns=['Values']),
            order=(1, 0, 1),
            seasonal_order=(0, 1, 1, 12)
        )
        model_fit = model.fit(disp=False)

        forecast = model_fit.get_forecast(steps=26, exog=station_data.drop(columns=['Values']).values[
                                                         -26:])
        forecast_values = forecast.predicted_mean
        forecasts_final[station] = forecast_values

    forecast_final_df = pd.DataFrame(forecasts_final)
    forecast_final_df.insert(0, 'date', pd.date_range(start='2022-01-01', end='2024-02-01', freq='MS'))
    csv_columns = pd.read_csv('datasets_ehyd/gw_test_empty.csv', nrows=0).columns.tolist()
    forecast_final_df = forecast_final_df[csv_columns]

    forecast_final_df.to_csv("forecast_final.csv", index=False)
    print("--------------------- forecast_final.csv: Complete")

if __name__ == "__main__":
    main()