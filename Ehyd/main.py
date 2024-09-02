# IMPORTS
import os
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', category=ConvergenceWarning)
import pandas as pd
from datetime import datetime
from scipy.spatial import distance
from collections import Counter
import seaborn as sns
import itertools
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 500)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense



# FUNCTIONS
def station_coordinates(input):
    """
    Creates a dataset consisting of measurement station IDs and their corresponding X and Y coordinates.

    Args:
        input: Directory of the measurement station CSV file.

    Returns:
        df: A DataFrame containing columns "x", "y", and "hzbnr01".
    """
    df = pd.read_csv(os.path.join("Ehyd", "datasets_ehyd", input, "messstellen_alle.csv"), sep=";")
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

                                value_str = value_str.strip().replace('Lücke', 'NaN')  # Convert 'Lücke' to NaN

                                # Skip rows with invalid data or specific keywords
                                if any(keyword in value_str for keyword in ["F", "K", "rekonstruiert aus Version 3->"]):
                                    continue

                                # Convert value to float
                                try:
                                    value = np.float32(value_str.replace(',', '.'))
                                except ValueError:
                                    continue

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
    for df_name, df_value in df_dict.items():
        # Tarih sütununu datetime format?na çevir
        df_value['Date'] = pd.to_datetime(df_value['Date'])

        # Günlük verileri ayl?k veriye dönü?tür
        if df_value['Date'].dt.to_period('D').nunique() > df_value['Date'].dt.to_period('M').nunique():
            df_value.set_index('Date', inplace=True)
            df_dict[df_name] = df_value.resample('MS').mean()  # MS: Ay ba??

        # Ayl?k veriler için sadece indeks ayarla
        else:
            df_value.set_index('Date', inplace=True)
            df_dict[df_name] = df_value

        # Mevcut indeksi yeniden indeksle ve join yöntemiyle birle?tir
        all_dates = pd.date_range(start='1960-01-01', end='2021-12-01', freq='MS')
        new_df = pd.DataFrame(index=all_dates)
        df_dict[df_name] = new_df.join(df_dict[df_name], how='left').fillna("NaN")

    return df_dict

def filter_dataframes_by_points(dataframes_dict, points_list):
    """
    Filters a dictionary of DataFrames to include only those whose names are specified in a given CSV file.

    Args:
        dataframes_dict (dict): A dictionary where keys are names (str) and values are DataFrames.
        points_list (str): Path to a CSV file that contains the names (str) of the DataFrames to filter.

    Returns:
        dict: A filtered dictionary containing only the DataFrames whose names are listed in the CSV file.
    """
    filtered_dict = {name: df for name, df in dataframes_dict.items() if name in points_list}
    return filtered_dict

#####################################
# Creating Dataframes from given CSVs
#####################################

"""
Buraday? k?salt?yorum hemen a?a??da 
##################################### Groundwater
groundwater_all_coordinates = station_coordinates("Groundwater")

# Groundwater Level
groundwater_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Groundwater", "Grundwasserstand-Monatsmittel")
groundwater_dict, groundwater_coordinates = to_dataframe(groundwater_folder_path, groundwater_all_coordinates)
groundwater_dict = process_dataframes(groundwater_dict)
to_global(groundwater_dict, prefix="gw_")

# Groundwater Temperature
groundwater_temperature_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Groundwater", "Grundwassertemperatur-Monatsmittel")
groundwater_temperature_dict, groundwater_temperature_coordinates = to_dataframe(groundwater_temperature_folder_path, groundwater_all_coordinates)
groundwater_temperature_dict = process_dataframes(groundwater_temperature_dict)
to_global(groundwater_temperature_dict, prefix="gwt_")

# Creating new dictionaries according to requested stations
points = pd.read_csv(os.path.join("Ehyd", "datasets_ehyd", "gw_test_empty.csv"))
points_list = [col for col in points.columns[1:]]

filtered_groundwater_dict = filter_dataframes_by_points(groundwater_dict, points_list)
filtered_gw_coordinates = groundwater_coordinates[groundwater_coordinates['hzbnr01'].isin([int(i) for i in points_list])]

##################################### Precipitation
precipitation_coordinates = station_coordinates("Precipitation")

# Rain
rain_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Precipitation", "N-Tagessummen")
rain_dict, rain_coordinates = to_dataframe(rain_folder_path, precipitation_coordinates)
rain_dict_monthly = process_dataframes(rain_dict)
to_global(rain_dict_monthly, prefix="rain_")

# Snow
snow_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Precipitation", "NS-Tagessummen")
snow_dict, snow_coordinates = to_dataframe(snow_folder_path, precipitation_coordinates)
snow_dict = process_dataframes(snow_dict)
to_global(snow_dict, prefix="snow_")

##################################### Sources
sources_coordinates = station_coordinates("Sources")

# Flow Rate
source_flow_rate_path = os.path.join("Ehyd", "datasets_ehyd", "Sources", "Quellschüttung-Tagesmittel")
source_flow_rate_dict, source_flow_rate_coordinates = to_dataframe(source_flow_rate_path, sources_coordinates)
source_flow_rate_dict_monthly = process_dataframes(source_flow_rate_dict)
to_global(source_flow_rate_dict_monthly, prefix="source_fr_")

# Conductivity
conductivity_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Sources", "Quellleitfähigkeit-Tagesmittel")
conductivity_dict, conductivity_coordinates = to_dataframe(conductivity_folder_path, sources_coordinates)
conductivity_dict_monthly = process_dataframes(conductivity_dict)
to_global(conductivity_dict_monthly, prefix="conductivity_")

# Source Temperature
source_temp_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Sources", "Quellwassertemperatur-Tagesmittel")
source_temp_dict, source_temp_coordinates = to_dataframe(source_temp_folder_path, sources_coordinates)
source_temp_dict_monthly = process_dataframes(source_temp_dict)
to_global(source_temp_dict_monthly, prefix="source_temp_")

##################################### Surface Water

surface_water_coordinates = station_coordinates("Surface_Water")

# Surface Water Level
surface_water_level_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", "W-Tagesmittel")
surface_water_level_dict, surface_water_level_coordinates = to_dataframe(surface_water_level_folder_path, surface_water_coordinates)
surface_water_level_dict_monthly = process_dataframes(surface_water_level_dict)
to_global(surface_water_level_dict_monthly, prefix="surface_water_level")

# Surface Water Temperature
surface_water_temp_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", "WT-Monatsmittel")
surface_water_temp_dict, surface_water_temp_coordinates = to_dataframe(surface_water_temp_folder_path, surface_water_coordinates)
surface_water_temp_dict = process_dataframes(surface_water_temp_dict)
to_global(surface_water_temp_dict, prefix="surface_water_temp")

# Sediment
sediment_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", "Schwebstoff-Tagesfracht")
sediment_dict, sediment_coordinates = to_dataframe(sediment_folder_path, surface_water_coordinates)  # daily version
sediment_dict = process_dataframes(sediment_dict)
to_global(sediment_dict, prefix="sediment_")

# Surface Water Flow Rate
surface_water_flow_rate_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", "Q-Tagesmittel")
surface_water_flow_rate_dict, surface_water_flow_rate_coordinates = to_dataframe(surface_water_flow_rate_folder_path, surface_water_coordinates)
surface_water_flow_rate_dict_monthly = process_dataframes(surface_water_flow_rate_dict)
to_global(surface_water_flow_rate_dict_monthly, prefix="surface_water_fr_")


# geçici pkl
# write pickle
with open('sediment_dict.pkl', 'wb') as f:
    pickle.dump(sediment_dict, f)

with open('surface_water_level_dict_monthly.pkl', 'wb') as f:
    pickle.dump(surface_water_level_dict_monthly, f)

with open('surface_water_flow_rate_dict_monthly.pkl', 'wb') as f:
    pickle.dump(surface_water_flow_rate_dict_monthly, f)

with open('surface_water_temp_dict.pkl', 'wb') as f:
    pickle.dump(surface_water_temp_dict, f)

with open('filtered_groundwater_dict.pkl', 'wb') as f:
    pickle.dump(filtered_groundwater_dict, f)

with open('snow_dict.pkl', 'wb') as f:
    pickle.dump(snow_dict, f)

with open('conductivity_dict.pkl', 'wb') as f:
    pickle.dump(conductivity_dict_monthly, f)

with open('source_flow_rate_dict.pkl', 'wb') as f:
    pickle.dump(source_flow_rate_dict, f)

with open('source_temp_dict.pkl', 'wb') as f:
    pickle.dump(source_temp_dict, f)

with open('rain_dict.pkl', 'wb') as f:
    pickle.dump(rain_dict, f)
    
"""
def process_and_store_data(folder, coordinates, prefix, points_list=None):
    data_dict, data_coordinates = to_dataframe(folder, coordinates)
    data_dict = process_dataframes(data_dict)
    to_global(data_dict, prefix=prefix)

    if points_list:
        data_dict = filter_dataframes_by_points(data_dict, points_list)

    return data_dict, data_coordinates

def save_to_pickle(data_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)

# Define paths and coordinates
groundwater_all_coordinates = station_coordinates("Groundwater")
precipitation_coordinates = station_coordinates("Precipitation")
sources_coordinates = station_coordinates("Sources")
surface_water_coordinates = station_coordinates("Surface_Water")

# Groundwater Temperature Dictionary
gw_folders = [("Groundwater/Grundwassertemperatur-Monatsmittel", "gwt_")]
for folder, prefix in gw_folders:
    gw_temp_dict, gw_temp_coordinates = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", folder), groundwater_all_coordinates, prefix)

# Filtered Groundwater Dictionary
points = pd.read_csv(os.path.join("Ehyd", "datasets_ehyd", "gw_test_empty.csv"))
points_list = [col for col in points.columns[1:]]
filtered_groundwater_dict, filtered_gw_coordinates = process_and_store_data(
    os.path.join("Ehyd", "datasets_ehyd", "Groundwater/Grundwasserstand-Monatsmittel"),
    groundwater_all_coordinates, "gw_", points_list)

# Precipitation: Rain and Snow
precipitation_folders = [
    ("Precipitation/N-Tagessummen", "rain_"),
    ("Precipitation/NS-Tagessummen", "snow_")]
for folder, prefix in precipitation_folders:
    dict_name, dict_coord = f"{prefix}_dict", f"{prefix}_coordinates"
    dict_name, dict_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", folder), precipitation_coordinates, prefix)

######### gizmo
for folder, prefix in precipitation_folders:
    dict_name, dict_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", folder), precipitation_coordinates, prefix)
    globals()[f"{prefix}_dict"] = dict_name
    globals()[f"{prefix}_coordinates"] = dict_coord
##########
def create_dict(main_folder, coordinates):
    for folder, prefix in main_folder:
        dict_name, dict_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", folder), coordinates, prefix)
        globals()[f"{prefix}_dict"] = dict_name
        globals()[f"{prefix}_coordinates"] = dict_coord

create_dict(precipitation_folders, precipitation_folders)
create_dict(source_folders, source_folders)
create_dict(source_folders, )


# Sources: Flow Rate, Conductivity, Temperature
source_folders = [
    ("Sources/Quellschüttung-Tagesmittel", "source_fr_"),
    ("Sources/Quellleitfähigkeit-Tagesmittel", "conductivity_"),
    ("Sources/Quellwassertemperatur-Tagesmittel", "source_temp_")]
for folder, prefix in source_folders:
    process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", folder), sources_coordinates, prefix)

# gizmo
for folder, prefix in source_folders:
    dict_name, dict_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", folder), sources_coordinates, prefix)
    globals()[f"{prefix}_dict"] = dict_name
    globals()[f"{prefix}_coordinates"] = dict_coord

# Surface Water: Level, Temperature, Sediment, Flow Rate
surface_water_folders = [
    ("Surface_Water/W-Tagesmittel", "surface_water_level_"),
    ("Surface_Water/WT-Monatsmittel", "surface_water_temp_"),
    ("Surface_Water/Schwebstoff-Tagesfracht", "sediment_"),
    ("Surface_Water/Q-Tagesmittel", "surface_water_fr_")]
for folder, prefix in surface_water_folders:
    process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", folder), surface_water_coordinates, prefix)

# gizmo
for folder, prefix in surface_water_folders:
    dict_name, dict_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", folder), surface_water_coordinates, prefix)
    globals()[f"{prefix}_dict"] = dict_name
    globals()[f"{prefix}_coordinates"] = dict_coord

# Save data to pickle files
pickle_files = {
    'gw_temp_dict.pkl': gw_temp_dict,
    'filtered_groundwater_dict.pkl': filtered_groundwater_dict,
    'snow_dict.pkl': snow_dict,
    'rain_dict.pkl': rain_dict,
    'surface_water_level_dict.pkl': surface_water_level_dict, #burada "monthly" vard?, sildim
    'surface_water_flow_rate_dict.pkl': surface_water_fdictlow_rate_dict, #burada "monthly" vard?, sildim
    'surface_water_temp_dict.pkl': surface_water_temp_dict,
    'conductivity_dict.pkl': conductivity_dict ,#burada sadece sa?da "monthly" vard?, sildim
    'source_flow_rate_dict.pkl': source_flow_rate_dict,
    'source_temp_dict.pkl': source_temp_dict}

for filename, data_dict in pickle_files.items():
    save_to_pickle(data_dict, filename)


########################################################################################################################
# Gathering associated features for 487 stations
########################################################################################################################

def calculate_distance(coord1, coord2):
    return distance.euclidean(coord1, coord2)

def find_nearest_coordinates(gw_row, df, k=20):
    distances = df.apply(lambda row: calculate_distance(
        (gw_row['x'], gw_row['y']),
        (row['x'], row['y'])
    ), axis=1)
    nearest_indices = distances.nsmallest(k).index
    return df.loc[nearest_indices]

# Creating a dataframe that stores all the associated features of the 487 stations.
data = pd.DataFrame()
def add_nearest_coordinates_column(df_to_add, name, k, df_to_merge=None):
    if df_to_merge is None:
        df_to_merge = data  # Use the current value of 'data' as the default
    results = []

    # Find the nearest points according to the coordinates
    for _, gw_row in filtered_gw_coordinates.iterrows():
        nearest = find_nearest_coordinates(gw_row, df_to_add, k)
        nearest_list = nearest['hzbnr01'].tolist()
        results.append({
            'hzbnr01': gw_row['hzbnr01'],
            name: nearest_list
        })

    results_df = pd.DataFrame(results)

    # Debug: Check if 'hzbnr01' exists in both dataframes
    print("Columns in df_to_merge:", df_to_merge.columns)
    print("Columns in results_df:", results_df.columns)

    # Ensure that the column exists in both dataframes before merging
    if 'hzbnr01' in df_to_merge.columns and 'hzbnr01' in results_df.columns:
        # Merge operation
        df = df_to_merge.merge(results_df, on='hzbnr01', how='inner')

        # Debug: Birle?tirilmi? DataFrame'i yazd?rarak kontrol et
        print("Merged DataFrame:")
        print(df.head())
    else:
        raise KeyError("Column 'hzbnr01' does not exist in one of the dataframes.")

    return df

data = add_nearest_coordinates_column(groundwater_temperature_coordinates, 'nearest_gw_temp', 1, df_to_merge=filtered_gw_coordinates)
data = add_nearest_coordinates_column(rain_coordinates, 'nearest_rain', 3, df_to_merge=data) # TODO burada data arguman? default oldugu icin silebiliriz.
data = add_nearest_coordinates_column(snow_coordinates, 'nearest_snow', 3, df_to_merge=data)
data = add_nearest_coordinates_column(source_flow_rate_coordinates, 'nearest_source_fr', 1, df_to_merge=data)
data = add_nearest_coordinates_column(conductivity_coordinates, 'nearest_conductivity', 1, df_to_merge=data)
data = add_nearest_coordinates_column(source_temp_coordinates, 'nearest_source_temp', 1, df_to_merge=data)
data = add_nearest_coordinates_column(surface_water_level_coordinates, 'nearest_owf_level', 3, df_to_merge=data)
data = add_nearest_coordinates_column(surface_water_temp_coordinates, 'nearest_owf_temp', 1, df_to_merge=data)
data = add_nearest_coordinates_column(sediment_coordinates, 'nearest_sediment', 1, df_to_merge=data)
data = add_nearest_coordinates_column(surface_water_flow_rate_coordinates, 'nearest_owf_fr', 3, df_to_merge=data)
data.drop(["x", "y"], axis=1, inplace=True)

data.to_csv('data.csv', index=False)

###################################################################################
# yer alt? suyu s?cakl?k tur?usu için data datframe'ine göre azaltma i?lemleri
################################################################################
data['nearest_gw_temp'].explode().nunique()  # 276

# data_gw_temp_dict isimli yeni sözlü?ü olu?turuyoruz
data_gw_temp_dict = {}

# nearest_gw_temp kolonundaki her bir listeyi döngü ile al?yoruz
for key_list in data['nearest_gw_temp']:
    # Liste içindeki her bir de?eri string'e çevirip kontrol ediyoruz
    for key in key_list:
        str_key = str(key)
        # E?er str_key groundwater_temperature_dict'te varsa, bu key ve ilgili dataframe'i yeni sözlü?e ekle
        if str_key in groundwater_temperature_dict:
            data_gw_temp_dict[str_key] = groundwater_temperature_dict[str_key]

len(data_gw_temp_dict)  # 276

with open('data_gw_temp_dict.pkl', 'wb') as f:
    pickle.dump(data_gw_temp_dict, f)


# buradayd?k - pickle'dan çekme
pkl_files = [f for f in os.listdir() if f.endswith('.pkl')]

for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as file:
        var_name = pkl_file[:-4]
        globals()[var_name] = pickle.load(file)


########################################################################################################################
# Imputing NaN Values
########################################################################################################################
def nan_imputer(dict):
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

#########
filled_data_gw_temp_dict = nan_imputer(data_gw_temp_dict)
filled_conductivity_dict = nan_imputer(conductivity_dict_monthly)
filled_source_flow_rate_dict = nan_imputer(source_flow_rate_dict_monthly)
filled_source_temp_dict = nan_imputer(source_temp_dict_monthly)
filled_rain_dict = nan_imputer(rain_dict_monthly)
filled_data_gw_temp_dict = nan_imputer(data_gw_temp_dict)
filled_conductivity_dict = nan_imputer(conductivity_dict_monthly)
filled_source_flow_rate_dict = nan_imputer(source_flow_rate_dict_monthly)
filled_source_temp_dict = nan_imputer(source_temp_dict_monthly)
filled_rain_dict = nan_imputer(rain_dict_monthly)


# write pickle
with open('filled_sediment_dict.pkl', 'wb') as f:
    pickle.dump(filled_sediment_dict, f)

with open('filled_surface_water_level_dict_monthly.pkl', 'wb') as f:
    pickle.dump(filled_surface_water_level_dict_monthly, f)

with open('filled_surface_water_flow_rate_dict_monthly.pkl', 'wb') as f:
    pickle.dump(filled_surface_water_flow_rate_dict_monthly, f)

with open('filled_surface_water_temp_dict.pkl', 'wb') as f:
    pickle.dump(filled_surface_water_temp_dict, f)

with open('filled_groundwater_dict.pkl', 'wb') as f:
    pickle.dump(filled_groundwater_dict, f)

with open('filled_snow_dict_monthly.pkl', 'wb') as f:
    pickle.dump(filled_snow_dict_monthly, f)

with open('filled_data_gw_temp_dict.pkl', 'wb') as f:
    pickle.dump(filled_data_gw_temp_dict, f)

with open('filled_conductivity_dict.pkl', 'wb') as f:
    pickle.dump(filled_conductivity_dict, f)

with open('filled_source_flow_rate_dict.pkl', 'wb') as f:
    pickle.dump(filled_source_flow_rate_dict, f)

with open('filled_source_temp_dict.pkl', 'wb') as f:
    pickle.dump(filled_source_temp_dict, f)

with open('filled_rain_dict.pkl', 'wb') as f:
    pickle.dump(filled_rain_dict, f)



###############################################################################################################
###############################################################################################################
############################# TURSUDAN SONRASINI ÇALI?TIRAB?L?R?Z #############################################
###############################################################################################################
###############################################################################################################
# Pickle dosyalar?n? toplu açma i?lemi:
pkl_files = [f for f in os.listdir() if f.endswith('.pkl')]

for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as file:
        var_name = pkl_file[:-4]
        globals()[var_name] = pickle.load(file)

data = pd.read_csv("data.csv")

#######################################################################

# 0. lstm'ye girecek ayl?k dataseti iskeletlerini haz?rla
# 1. bu a?a??dakileri dene,
# 2. yeralt? su seviyeleri ile korelasyonlar?na bak
# 3. her noktan?n en iyi korele oldu?u sütunu seç
# 4. lstm'ye girecek final verisetine o sütunu koy ya?mur için.
# 5. bunu kar, sediment vs için de yap.

#######################################################################
# sözlüklerin içindeki serileri dataframe yap?yorum.
dict_list = [filled_conductivity_dict, filled_data_gw_temp_dict, filled_groundwater_dict, filled_rain_dict,
             filled_sediment_dict, filled_snow_dict_monthly, filled_source_flow_rate_dict,
             filled_source_temp_dict, filled_surface_water_flow_rate_dict_monthly,
             filled_surface_water_level_dict_monthly, filled_surface_water_temp_dict]


# Dict list içindeki her bir sözlü?ün value'lar?n? DataFrame yapacak fonksiyon
def convert_series_to_dataframe(d):
    for key in d:
        d[key] = d[key].to_frame(name=key)
    return d

for i in range(len(dict_list)):
    dict_list[i] = convert_series_to_dataframe(dict_list[i])

# TODO SARIMA gerek
# hangi sözlükte kaç? 2021 aral?kta bitmiyor:
mapping_dict = {
    'hzbnr01': filled_groundwater_dict,
    'nearest_gw_temp': filled_data_gw_temp_dict,
    'nearest_rain': filled_rain_dict,
    'nearest_snow': filled_snow_dict_monthly,
    'nearest_source_fr': filled_source_flow_rate_dict,
    'nearest_conductivity': filled_conductivity_dict,
    'nearest_source_temp': filled_source_temp_dict,
    'nearest_owf_level': filled_surface_water_level_dict_monthly,
    'nearest_owf_temp': filled_surface_water_temp_dict,
    'nearest_sediment': filled_sediment_dict_monthly,
    'nearest_owf_fr': filled_surface_water_flow_rate_dict_monthly,
}

# Son indeks 2021 Aral?k olmayan DataFrame'lerin say?s?n? saklayacak bir sözlük olu?turun
non_dec_2021_counts = {key: 0 for key in mapping_dict.keys()}

##########
# Mevcut olan kodlar? saklamak için sözlükleri ba?lat
existing_codes_dict = {}

# Tüm sözlükleri döngüye al
for key, df_dict in mapping_dict.items():
    # Listeyi ba?lat
    key_list = []

    for code, df in df_dict.items():
        if not df.empty:
            # ?ndeksi datetime format?na dönü?tür
            df.index = pd.to_datetime(df.index, errors='coerce')
            last_index = df.index[-1]

            if last_index is not pd.NaT and not (last_index.year == 2021 and last_index.month == 12):
                non_dec_2021_counts[key] += 1
                print(f"{key} - Code: {code}, Last Index: {last_index}, 2021 Aral?k de?il")
                key_list.append(str(code))  # code'u string olarak ekle
        else:
            print(f"{key} - Code: {code} DataFrame is empty")

    # Benzersiz kodlar? elde et
    key_list = list(set(key_list))

    if key_list:
        globals()[f"{key}_list"] = key_list

# Mevcut olan kodlar? 'data' DataFrame'inde kontrol edip sözlüklerde sakla
for key in mapping_dict.keys():
    list_name = f"{key}_list"
    if list_name in globals():
        current_list = globals()[list_name]
        print(f"\n{list_name}: {current_list}")
        print(f"{list_name} uzunlu?u: {len(current_list)}")

        # 'data' DataFrame'indeki ilgili sütunu kontrol et
        if key in data.columns:
            existing_codes_dict[key] = []
            for code in current_list:
                if data[key].str.contains(code).any():
                    print(f"{code} {key} sütununda mevcut.")
                    existing_codes_dict[key].append(code)  # Mevcut olan kodu sözlü?e ekle
                else:
                    print(f"{code} {key} sütununda bulunamad?.")
            # Mevcut olan kod listesinin uzunlu?unu yazd?r
            print(f"Mevcut olan kodlar?n uzunlu?u {key}: {len(existing_codes_dict[key])}")
        else:
            print(f"{key} isimli sütun 'data' DataFrame'inde mevcut de?il.")
    else:
        print(f"{list_name} listesi bo? veya olu?turulmad?.")

# Sonuçlar? yazd?r
print("\nMevcut olan kodlar:")
for key, codes in existing_codes_dict.items():
    print(f"{key}: {codes}")
    print(f"{key} için mevcut kod say?s?: {len(codes)}")
#########
# burada existing_codes_dict sözlü?ündeki dataframeleri Sarima ile 2021e kadar olmayanlar? doldurmay? deneyelim:

def sarima_forecast_for_nan(mapping_dict, existing_codes_dict):
    """
    Bu fonksiyon, mapping_dict içindeki her DataFrame için,
    existing_codes_dict'teki ID'lerle e?le?en DataFrame'lerin ba?l?klar?n? yazd?r?r
    ve her key için kaç adet head yazd?r?ld???n? bildirir.

    :param mapping_dict: Anahtarlar?n DataFrame'lere e?lendi?i sözlük
    :param existing_codes_dict: Her anahtar için ID listelerinin bulundu?u sözlük
    """
    total_counts = []  # Her bir key için yazd?r?lan head say?s?n? tutacak liste

    for key, data_dict in mapping_dict.items():
        count = 0  # Her key için yazd?r?lan head say?s?n? takip etmek için sayaç

        # E?er data_dict bir sözlükse (DataFrame'lerin bulundu?u sözlük)
        if isinstance(data_dict, dict):
            ids_list = existing_codes_dict.get(key, [])
            for data_id in ids_list:
                # ID'nin anahtar olarak bulundu?u DataFrame'i al
                if data_id in data_dict:
                    df = data_dict[data_id]
                    # DataFrame'in head'ini yazd?r
                    print(f"Head of DataFrame for ID {data_id} in '{key}':")
                    print(df.head())
                    print("\n")  # Her bir DataFrame aras?na bo?luk ekler
                    count += 1  # Her yazd?r?lan head için sayac? art?r
                else:
                    print(f"ID {data_id} not found in the dictionary for key '{key}'.")
        else:
            print(f"Value for key '{key}' is not a dictionary or DataFrame.")

        # Her bir key için kaç adet head yazd?r?ld???n? listeye ekle
        total_counts.append(f"{key}: {count}")

    # Sonuçlar? tek sat?rda yazd?r
    print("Total heads printed per key: " + ", ".join(total_counts))


# Fonksiyonu kullanmak için:
sarima_forecast_for_nan(mapping_dict, existing_codes_dict)



###################333
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

for key, value in filled_groundwater_dict.items():
    print(value[value == 0].count())

nani = filled_groundwater_dict["324095"]
nani[nani == 0].count()


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


# 11 sözlükteki tüm dataframe'lerin tüm kolonlar?n? float32'ye çevirme
def convert_to_float32(df):
    return df.astype('float32')

# Her bir sözlükteki veri çerçevelerini veri tipini float32'ye çevirme
for dictionary in dict_list:
    for key in dictionary:
        # Veri tipini float32'ye çevir
        dictionary[key] = convert_to_float32(dictionary[key])


################ 733 dataframe
list_of_dfs = []

# 733 ay için döngü
for month in range(720):

    index_values = data['hzbnr01']
    new_df = pd.DataFrame(index=index_values)

    # gw
    new_df['gw_level'] = None
    new_df['gw_level_lag_1'] = None
    new_df['gw_level_rolling_mean_6_lag_1'] = None

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

    # birliler
    def for_one_point(new_df, data, filled_dict, nearest_column, variable_type):
        # Yeni kolonlar? ba?lat?yoruz
        new_df[f'{variable_type}'] = None
        new_df[f'{variable_type}_lag_1'] = None
        new_df[f'{variable_type}_rolling_mean_6_lag_1'] = None

        # Tüm indekslerde dola??yoruz
        for index in new_df.index:
            # data DataFrame'inde hzbnr01 ile e?le?en nearest_column listesini al?yoruz
            nearest_list_str = data.loc[data['hzbnr01'] == index, nearest_column].values

            # E?er nearest_list_str bo? de?ilse
            if len(nearest_list_str) > 0:
                # nearest_column listesinin string de?erini gerçek listeye dönü?türüyoruz
                nearest_list = ast.literal_eval(nearest_list_str[0])

                # nearest_column listesinin ilk eleman?n? al?yoruz
                if len(nearest_list) > 0:
                    str_index = str(nearest_list[0])  # Tek eleman? string'e çeviriyoruz

                    # Eleman için
                    if str_index in filled_dict:
                        relevant_df = filled_dict[str_index]
                        if len(relevant_df) > 0:
                            new_df.at[index, f'{variable_type}'] = relevant_df.iloc[
                                0, 0]  # ?lk sat?r?n ilk kolon de?eri
                        if len(relevant_df.columns) > 1:
                            new_df.at[index, f'{variable_type}_lag_1'] = relevant_df.iloc[
                                0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                        if len(relevant_df.columns) > 2:
                            new_df.at[index, f'{variable_type}_rolling_mean_6_lag_1'] = relevant_df.iloc[
                                0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
                    else:
                        print(f"Warning: Key '{str_index}' not found in {variable_type} dictionary")
            else:
                print(f"Warning: No {nearest_column} list found for index '{index}'")

        return new_df

    new_df = for_one_point(new_df, data, filled_data_gw_temp_dict, 'nearest_gw_temp', 'gw_temp')
    new_df = for_one_point(new_df, data, filled_conductivity_dict, 'nearest_conductivity', 'conductivity')
    new_df = for_one_point(new_df, data, filled_source_flow_rate_dict, 'nearest_source_fr', 'source_fr')
    new_df = for_one_point(new_df, data, filled_source_temp_dict, 'nearest_source_temp', 'source_temp')
    new_df = for_one_point(new_df, data, filled_surface_water_temp_dict, 'nearest_owf_temp', 'owf_temp')
    new_df = for_one_point(new_df, data, filled_sediment_dict_monthly, 'nearest_sediment', 'sediment')

    # üçlüler
    def for_three_points(new_df, data, filled_dict, nearest_column, variable_type):
        # Yeni kolonlar ba?lat?l?yor
        for i in range(1, 4):
            new_df[f'{variable_type}_{i}'] = None
            new_df[f'{variable_type}_{i}_lag_1'] = None
            new_df[f'{variable_type}_{i}_rolling_mean_6_lag_1'] = None

        # Tüm indekslerde dola??yoruz
        for index in new_df.index:
            # data DataFrame'inde hzbnr01 ile e?le?en nearest_column listesini al?yoruz
            nearest_list_str = data.loc[data['hzbnr01'] == index, nearest_column].values

            # E?er nearest_list_str bo? de?ilse
            if len(nearest_list_str) > 0:
                # nearest_column listesinin string de?erini gerçek listeye dönü?türüyoruz
                nearest_list = ast.literal_eval(nearest_list_str[0])

                # nearest_column listesinin ilk üç eleman?n? al?yoruz
                for i in range(min(3, len(nearest_list))):
                    str_index = str(nearest_list[i])  # Eleman? string'e çeviriyoruz

                    # Eleman için
                    if str_index in filled_dict:
                        relevant_df = filled_dict[str_index]
                        if len(relevant_df) > 0:
                            new_df.at[index, f'{variable_type}_{i + 1}'] = relevant_df.iloc[
                                0, 0]  # ?lk sat?r?n ilk kolon de?eri
                        if len(relevant_df.columns) > 1:
                            new_df.at[index, f'{variable_type}_{i + 1}_lag_1'] = relevant_df.iloc[
                                0, 1]  # ?lk sat?r?n ikinci kolon de?eri
                        if len(relevant_df.columns) > 2:
                            new_df.at[index, f'{variable_type}_{i + 1}_rolling_mean_6_lag_1'] = relevant_df.iloc[
                                0, 2]  # ?lk sat?r?n üçüncü kolon de?eri
                    else:
                        print(f"Warning: Key '{str_index}' not found in {variable_type} dictionary")
            else:
                print(f"Warning: No {nearest_column} list found for index '{index}'")

        return new_df


    new_df = for_three_points(new_df, data, filled_rain_dict, 'nearest_rain', 'rain')
    new_df = for_three_points(new_df, data, filled_snow_dict_monthly, 'nearest_snow', 'snow')
    new_df = for_three_points(new_df, data, filled_surface_water_level_dict_monthly, 'nearest_owf_level', 'owf_level')
    new_df = for_three_points(new_df, data, filled_surface_water_flow_rate_dict_monthly, 'nearest_owf_fr', 'owf_fr')


    # Listeye ekliyoruz
    list_of_dfs.append(new_df)

# list_of_dfs tur?usu kuruldu
with open('list_of_dfs.pkl', 'wb') as f:
    pickle.dump(list_of_dfs, f)

# list_of_dfs tur?udan ç?karma i?lemi
with open('list_of_dfs.pkl', 'rb') as f:
    list_of_dfs = pickle.load(f)

################################ Normalizasyon
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_dfs = [pd.DataFrame(scaler.fit_transform(df), columns=df.columns) for df in list_of_dfs]


#### LSTM - omg it's happening
# 1. Veri Haz?rl???

# DataFrame'leri numpy array'lerine dönü?türüp birle?tirin
data = np.array([df.values for df in list_of_dfs])  # (720, 487, 57)

# 2. Pencereleme
def create_windows(data, window_size, forecast_horizon):
    X, y = [], []
    num_time_steps = data.shape[0]

    for start in range(num_time_steps - window_size - forecast_horizon + 1):
        end = start + window_size
        X.append(data[start:end, :, :])
        y.append(data[end:end + forecast_horizon, :, :])

    X = np.array(X)
    y = np.array(y)

    # X'in boyutlar?n? (batch_size, time_steps, features) haline getir
    X = X.reshape(X.shape[0], X.shape[1], -1)
    y = y.reshape(y.shape[0], y.shape[1], -1)

    return X, y


window_size = 12 # window size'? hala tam olarak anlayamad?m
forecast_horizon = 26 # önümüzdeli 26 ay? tahmin edece?iz
X, y = create_windows(data, window_size, forecast_horizon)  # X: (672, 12, 487, 5), y: (672, 26, 487, 5)

# 3. E?itim ve Test Setlerine Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. LSTM Modelini Olu?turma ve E?itim
model = Sequential()
model.add(LSTM(units=57, return_sequences=True, input_shape=(window_size, data.shape[1] * data.shape[2])))
model.add(LSTM(units=57, return_sequences=True))
model.add(Dense(data.shape[2]))  # Ç?k?? katman?, tahmin edilmesi gereken sütun say?s?na göre ayarlanmal?
model.compile(optimizer='adam', loss='mse')

# Modeli e?itme
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


