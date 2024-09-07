# IMPORTS
import tensorflow as tf

# inter_op ve intra_op paralellik ayarlar?
tf.config.threading.set_inter_op_parallelism_threads(4)  # Örne?in, 4 i? parçac??? kullan
tf.config.threading.set_intra_op_parallelism_threads(2)  # ??lem ba??na 2 i? parçac??? kullan
import os
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
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
warnings.simplefilter('ignore', category=ConvergenceWarning)


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

def save_to_pickle(item, filename):
    """
    Saves a dictionary to a pickle file.

    Args:
        data_dict (dict): The dictionary to save.
        filename (str): The path to the output pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(item, f)


########################################################################################################################
# Creating Dataframes from given CSVs
########################################################################################################################

# Define paths and coordinates
groundwater_all_coordinates = station_coordinates("Groundwater")
precipitation_coordinates = station_coordinates("Precipitation")
sources_coordinates = station_coordinates("Sources")
surface_water_coordinates = station_coordinates("Surface_Water")

# Precipitation: Rain and Snow
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

# Groundwater Dictionary (Filtered to the requested 487 stations)
stations = pd.read_csv(os.path.join("Ehyd", "datasets_ehyd", "gw_test_empty.csv"))
station_list = [col for col in stations.columns[1:]]
filtered_groundwater_dict, filtered_gw_coordinates = process_and_store_data(
    os.path.join("Ehyd", "datasets_ehyd", "Groundwater", "Grundwasserstand-Monatsmittel"),
    groundwater_all_coordinates, "gw_", station_list)

gw_temp_dict, gw_temp_coordinates = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Groundwater", "Grundwassertemperatur-Monatsmittel"), groundwater_all_coordinates, "gwt_")
rain_dict, rain_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Precipitation", precipitation_folders[0][0]), precipitation_coordinates, "rain_")
snow_dict, snow_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Precipitation", precipitation_folders[1][0]), precipitation_coordinates, "snow_")
source_fr_dict, source_fr_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Sources", source_folders[0][0]), sources_coordinates, "source_fr_")
conduct_dict, conduct_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Sources", source_folders[1][0]), sources_coordinates, "conduct_")
source_temp_dict, source_temp_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Sources", source_folders[2][0]), sources_coordinates, "source_temp_")
surface_water_lvl_dict, surface_water_lvl_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", surface_water_folders[0][0]), surface_water_coordinates, "surface_water_lvl_")
surface_water_temp_dict, surface_water_temp_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", surface_water_folders[1][0]), surface_water_coordinates, "surface_water_temp_")
sediment_dict, sediment_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", surface_water_folders[2][0]), surface_water_coordinates, "sediment_")
surface_water_fr_dict, surface_water_fr_coord = process_and_store_data(os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", surface_water_folders[3][0]), surface_water_coordinates, "surface_water_fr_")

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

    # Find the nearest stations according to the coordinates
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

data = add_nearest_coordinates_column(gw_temp_coordinates, 'nearest_gw_temp', 1, df_to_merge=filtered_gw_coordinates)
data = add_nearest_coordinates_column(rain_coord, 'nearest_rain', 3, df_to_merge=data) # TODO burada data arguman? default oldugu icin silebiliriz.
data = add_nearest_coordinates_column(snow_coord, 'nearest_snow', 3, df_to_merge=data)
data = add_nearest_coordinates_column(source_fr_coord, 'nearest_source_fr', 1, df_to_merge=data)
data = add_nearest_coordinates_column(conduct_coord, 'nearest_conductivity', 1, df_to_merge=data)
data = add_nearest_coordinates_column(source_temp_coord, 'nearest_source_temp', 1, df_to_merge=data)
data = add_nearest_coordinates_column(surface_water_lvl_coord, 'nearest_owf_level', 3, df_to_merge=data)
data = add_nearest_coordinates_column(surface_water_temp_coord, 'nearest_owf_temp', 1, df_to_merge=data)
data = add_nearest_coordinates_column(sediment_coord, 'nearest_sediment', 1, df_to_merge=data)
data = add_nearest_coordinates_column(surface_water_fr_coord, 'nearest_owf_fr', 3, df_to_merge=data)
data.drop(["x", "y"], axis=1, inplace=True)

file_path = os.path.join('Ehyd', 'pkl_files', 'data.pkl')
save_to_pickle(data, file_path)

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

########################################################################################################################
# Adding lagged values and rolling means
########################################################################################################################
filled_dict_list = [filled_gw_temp_dict, filled_filtered_groundwater_dict, filled_snow_dict, filled_rain_dict,
                    filled_conduct_dict, filled_source_fr_dict, filled_source_temp_dict, filled_surface_water_lvl_dict,
                    filled_surface_water_fr_dict, filled_surface_water_temp_dict, filled_sediment_dict]

def add_lag_and_rolling_mean(df, window=6):
    """
    Adds lagged and rolling mean columns to a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        window (int, optional): The window size for calculating the rolling mean. Default is 6.

    Returns:
        pandas.DataFrame: The DataFrame with additional columns for lagged values and rolling means.
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
# Zero Padding and changing the data type to float32
########################################################################################################################
for dictionary in filled_dict_list:
    for key, df in dictionary.items():
        df.fillna(0, inplace=True)
        df = df.astype(np.float32)
        dictionary[key] = df

# finalleri pickle'a alma
directory = 'Ehyd/pkl_files'

for dictionary in filled_dict_list:
    dict_name = [name for name in globals() if globals()[name] is dictionary][0]
    filename = os.path.join(directory, f'{dict_name}.pkl')
    save_to_pickle(dictionary, filename)

# Calling pickle files back from the directory
pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

for pkl_file in pkl_files:
    file_path = os.path.join(directory, pkl_file)
    with open(file_path, 'rb') as file:
        var_name = pkl_file[:-4]
        globals()[var_name] = pickle.load(file)

########################################################################################################################
# LSTM-formatted dataframes and the .pkl file
########################################################################################################################
data['hzbnr01'] = data['hzbnr01'].apply(lambda x: [x])

#### 487
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
# Her bir station için döngü
for idx, row in data.iterrows():
    # Kodlari liste içerisinden ç?kar
    code = str(row['hzbnr01'][0])

    # Sözlükten DataFrame'i al
    if code in filled_filtered_groundwater_dict:
        # Yeni DataFrame olu?tur
        df = filled_filtered_groundwater_dict[code].copy()

        for key, (prefix, source_dict) in data_sources.items():
            for i, code_value in enumerate(row[key]):
                code_str = str(code_value)
                source_df = source_dict.get(code_str, pd.DataFrame())
                # Yeni sütunlar? ekle
                source_df = source_df.rename(columns=lambda x: f'{prefix}_{i + 1}_{x}')
                df = df.join(source_df, how='left')

                columns = ["Values", "lag_1", "lag_2", "lag_3", "rolling_mean_original", "rolling_mean_6_lag_1", "rolling_mean_6_lag_2", "rolling_mean_6_lag_3"]
                for column in columns:
                    if i == 2:
                        df[f"{prefix}_{column}_mean"] = (df[f"{prefix}_{i + 1}_{column}"] + df[f"{prefix}_{i}_{column}"] + df[f"{prefix}_{i - 1}_{column}"]) / 3

        # Sonuçlar? sözlü?e ekle
        new_dataframes[code] = df

# 744
monthly_dataframes = {}
# Her y?l ve ay için döngü
for year in range(1960, 2022):  # örnek olarak 1960'dan 2024'e kadar
    for month in range(1, 13):  # 1'den 12'ye kadar
        # Y?l ve ay bilgisi ile olu?turulan anahtar
        key = f"{year}_{month:02d}"

        # Bo? bir liste olu?turup o y?l ve ay için verileri toplamak
        monthly_data = []

        for df_id, df in new_dataframes.items():
            # ?ndex'teki tarih bilgisi
            mask = (df.index.to_period("M").year == year) & (df.index.to_period("M").month == month)

            if mask.any():
                # ?lgili ay için filtrelenmi? veri
                filtered_df = df[mask]

                # ?ndeksleri güncelle
                new_index = [f"{df_id}" for i in range(len(filtered_df))]
                filtered_df.index = new_index

                monthly_data.append(filtered_df)

        # E?er bu y?l ve ay için veri varsa, bunlar? birle?tir ve yeni DataFrame olu?tur
        if monthly_data:
            # Her bir DataFrame'in ayn? sütun ve indekslere sahip oldu?unu varsayarak birle?tirme
            combined_df = pd.concat(monthly_data)

            # Sonuçlar? saklama
            monthly_dataframes[key] = combined_df


file_path = os.path.join("Ehyd", "pkl_files", 'monthly_dataframes.pkl')
save_to_pickle(monthly_dataframes, file_path)

"""
########################################################################################################################
# Normalization
########################################################################################################################
scaler = MinMaxScaler(feature_range=(0, 1))

# normalized_monthly_dfs_list = [pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns) for month, df in monthly_dataframes.items()]

normalized_dataframes = {}
for month, df in monthly_dataframes.items():
    normalized_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    normalized_dataframes[month] = normalized_df



for key, value in normalized_dataframes.items():
    print(value.tail())


# zero-padding kontrolü ve dataframe ay?klamas?
monthly_zero_ratio_dict = {}
value_count = (normalized_dataframes["1960_01"].shape[0] * normalized_dataframes["1960_01"].shape[1]) # 89608

for key, df in normalized_dataframes.items():
    zero_counts = (df == 0).sum().sum()
    zero_ratio = zero_counts / value_count
    monthly_zero_ratio_dict[key] = zero_ratio

keys = list(monthly_zero_ratio_dict.keys())
values = list(monthly_zero_ratio_dict.values())
last_value_above_threshold = next((val for val in reversed(values) if val > 0.5), None) # 0.501
key_of_last_value_above_threshold = keys[values.index(last_value_above_threshold)] # 1984_11

plt.figure(figsize=(10, 6))
plt.plot(keys, values)
plt.xlabel('Date (Year-Month)')
plt.ylabel('Zero Ratio')
plt.xticks(rotation=45)
plt.title('Zero Ratio over the Dataframes')
plt.tight_layout()
plt.show()

normalized_dict_85to21 = {k: v for k, v in normalized_dataframes.items() if k >= "1985_01"}
len(normalized_dict_85to21) # 444
normalized_dict_85to21["1985_01"].shape # (487, 184)

file_path = os.path.join("Ehyd", "pkl_files", 'normalized_dict_85to21.pkl')
save_to_pickle(normalized_dict_85to21, file_path)

# Dosyay? aç?n ve içeri?i yükleyin
with open(file_path, 'rb') as file:
    normalized_dict_85to21 = pickle.load(file)
"""

########################################################################################################################
# LSTM Model
########################################################################################################################
# 7 eylül Cumartesi

with open(os.path.join('Ehyd', 'pkl_files', 'monthly_dataframes.pkl'), 'rb') as file:
    monthly_dataframes = pickle.load(file)

monthly_dict_85to21 = {k: v for k, v in monthly_dataframes.items() if k >= "1985_01"}
len(monthly_dict_85to21)

file_path = os.path.join("Ehyd", "pkl_files", 'monthly_dict_85to21.pkl')
save_to_pickle(monthly_dict_85to21, file_path)

###############################################
# h?zland?r?c? faktörler deneme:
##############################################3
# Dosyay? aç?n ve içeri?i yükleyin
with open(os.path.join('Ehyd', 'pkl_files', 'monthly_dict_85to21.pkl'), 'rb') as file:
    monthly_dict_85to21 = pickle.load(file)


lookback = 36  # Modelin bakaca?? geçmi? ay say?s?
test_size = 24  # Tahmin edilecek 24 ay
batch_size = 256  # Bir seferde i?lenecek veri miktar?

X_train, y_train = [], []
X_test, y_test = [], []

# Her istasyon verisini e?itim ve test olarak ay?r?yoruz
for month, df in monthly_dict_85to21.items():
    values = df.values  # DataFrame'i numpy array'e çeviriyoruz

    # E?itim verisi: lookback ve test_size d???nda kalan veriler
    num_samples = len(values) - lookback - test_size
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        X_train_batch = np.array([values[i:i + lookback] for i in range(start, end)])
        y_train_batch = np.array([values[i + lookback:i + lookback + test_size, 0] for i in range(start, end)])

        # X_train ve y_train listesine batch'leri ekleme
        X_train.append(X_train_batch)
        y_train.append(y_train_batch)

    # Test verisi: Son 12 ay ve test verisi
    X_test.append(values[-(lookback + test_size):-test_size])  # Son 12 ayl?k geçmi? veriyi al?yoruz
    y_test.append(values[-test_size:, 0])  # Sonraki 24 ay?n target verisi

# X_train ve y_train'i numpy array'lere çeviriyoruz (batch'ler halinde)
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
X_test = np.array(X_test)
y_test = np.array(y_test)

####
print(type(X_train))  # This will confirm if it's a list
print(type(X_train[0]))  # This checks what type the first element is
print(X_train[:5])  # Print the first 5 elements to understand their structure
n_samples, n_timesteps, n_features = X_train.shape

import numpy as np

X_train = np.array(X_train)
print(X_train.shape)  # Ensure the shape is (888, n_features)
####

# Assuming X_train is a list of 3D arrays (n_timesteps, n_features)
scaler = MinMaxScaler()
scaled_X_train = []

# Loop over each 3D array in X_train
for X in X_train:
    # Get the shape of the current 3D array (n_samples, n_timesteps, n_features)
    n_samples, n_timesteps, n_features = X.shape
    # Reshape the data to 2D: (n_samples * n_timesteps, n_features)
    X_reshaped = X.reshape(-1, n_features)

    # Fit and transform the data
    X_scaled = scaler.fit_transform(X_reshaped)

    # Reshape back to the original shape: (n_samples, n_timesteps, n_features)
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)

    # Append the scaled array to the new list
    scaled_X_train.append(X_scaled)

scaled_y_train = []
for y in y_train:
    n_samples, n_timesteps, n_features = y.shape
    # Reshape the data to 2D: (n_samples * n_timesteps, n_features)
    y_reshaped = y.reshape(-1, n_features)

    # Fit and transform the data
    y_scaled = scaler.fit_transform(y_reshaped)

    # Reshape back to the original shape: (n_samples, n_timesteps, n_features)
    y_scaled = y_scaled.reshape(n_samples, n_timesteps, n_features)

    # Append the scaled array to the new list
    scaled_y_train.append(y_scaled)



###### dilara chatgpt ba?lang?c?
scaler = MinMaxScaler()
# Fit on the training data (for example, assuming X_train)
X_train_scaled = scaler.fit_transform(X_train)
# Transforming test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Later, to revert back the scaled data to original scale:
X_train_original = scaler.inverse_transform(X_train_scaled)
X_test_original = scaler.inverse_transform(X_test_scaled)

### biti?


############## gizmo
# Ayl?k verileri y?l baz?nda ay?rarak
train_data = {k: v for k, v in monthly_dict_85to21.items() if k < '2020_01'}
test_data = {k: v for k, v in monthly_dict_85to21.items() if k >= '2020_01'}



# E?itim setine scaler uygulama
scaler = MinMaxScaler()
#scaled_train_data = scaler.fit_transform(train_data)
# Test setine ayn? scaler'? uygulama
#scaled_test_data = scaler.transform(test_data)

scaled_train_data = []
# Loop through each item in the dictionary
for date, df in train_data.items():
    # Fit and transform the current data
    scaled_data = scaler.fit_transform(df)
    # Store the scaled data back in a new dictionary
    scaled_train_data.append(df) 

scaled_test_data = []
for date, df in test_data.items():
    # Fit and transform the current data
    scaled_data = scaler.transform(df)
    # Store the scaled data back in a new dictionary
    scaled_test_data.append(df)

X_train = scaled_train_data[:400]
y_train = scaled_train_data[400:]

X_test = scaled_test_data[:44]
y_test = scaled_test_data[44:]



scaled_train_data = np.array(scaled_train_data)






X_train, y_train, X_test, y_test

TRAIN A?AMASI                                   TEST A?AMASI
x_train (train snow bilmemne 2020te kadar)      x_test (rain snow bilmemne 2020den sonra)
y_train (gw 2020ye kadar)                       y_test (gw w2020den sonra)
                                                y_pred


num_features = X_train.shape[2]  # Kaç özellik (sütun) var?
num_units = 30  # LSTM hücre say?s?

model = Sequential()
model.add(LSTM(num_units, input_shape=(lookback, num_features), return_sequences=True))
model.add(Dropout(0.2))  # Overfitting'i önlemek için
model.add(LSTM(num_units, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(24))  # 24 ayl?k tahmin

model.compile(optimizer='adam', loss='mse')

# Early stopping: Do?rulama kayb? 5 dönem boyunca iyile?mezse e?itim duracak
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


import time

# E?itim süresini ölçelim
start_time = time.time()
# Modeli e?itme
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

end_time = time.time()

print(f"Training took {end_time - start_time} seconds")

# Test verisi üzerinde tahmin yapma
predictions = model.predict(X_test)

# Tahmin sonuçlar? ve gerçek sonuçlar? inceleme
print("Tahmin edilen de?erler: ", predictions[0:5])
print("Gerçek test de?erleri: ", y_test[0:5])







######### eda
lookback = 36  # Modelin bakaca?? geçmi? ay say?s?
test_size = 24  # Tahmin edilecek 24 ay
batch_size = 256  # Bir seferde i?lenecek veri miktar?

X_train, y_train = [], []
X_test, y_test = [], []

# Her istasyon verisini e?itim ve test olarak ay?r?yoruz
for month, df in monthly_dict_85to21.items():
    values = df.values  # DataFrame'i numpy array'e çeviriyoruz

    # E?itim verisi: lookback ve test_size d???nda kalan veriler
    num_samples = len(values) - lookback - test_size
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        X_train_batch = np.array([values[i:i + lookback] for i in range(start, end)])
        y_train_batch = np.array([values[i + lookback:i + lookback + test_size, 0] for i in range(start, end)])

        # X_train ve y_train listesine batch'leri ekleme
        X_train.append(X_train_batch)
        y_train.append(y_train_batch)

    # Test verisi: Son 36 ay ve test verisi
    X_test.append(values[-(lookback + test_size):-test_size])  # Son 36 ayl?k geçmi? veriyi al?yoruz
    y_test.append(values[-test_size:, 0])  # Sonraki 24 ay?n target verisi

# X_train ve y_train'i numpy array'lere çeviriyoruz
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Veriyi normalle?tirme (MinMaxScaler kullanarak)
scaler = MinMaxScaler()

# X_train verisi için normalle?tirme
scaled_X_train = []

# Her bir örnek (3D array) için normalizasyon yap?yoruz
for X in X_train:
    n_timesteps, n_features = X.shape  # ?u an (lookback, n_features) ?eklinde
    X_reshaped = X.reshape(-1, n_features)  # Veriyi (timesteps, features) olarak 2D'ye indiriyoruz
    X_scaled = scaler.fit_transform(X_reshaped)  # Veriyi normalle?tiriyoruz
    X_scaled = X_scaled.reshape(n_timesteps, n_features)  # Tekrar orijinal boyutuna döndürüyoruz
    scaled_X_train.append(X_scaled)

# X_test verisi için normalle?tirme
scaled_X_test = []

for X in X_test:
    n_timesteps, n_features = X.shape  # ?u an (lookback, n_features) ?eklinde
    X_reshaped = X.reshape(-1, n_features)  # Veriyi (timesteps, features) olarak 2D'ye indiriyoruz
    X_scaled = scaler.transform(X_reshaped)  # Test verisine ayn? scaler ile transform yap?yoruz
    X_scaled = X_scaled.reshape(n_timesteps, n_features)  # Tekrar orijinal boyutuna döndürüyoruz
    scaled_X_test.append(X_scaled)

# scaled_X_train ve scaled_X_test numpy array'e çevirme
scaled_X_train = np.array(scaled_X_train)
scaled_X_test = np.array(scaled_X_test)

# Veriyi modele beslemeye haz?r hale getirdik
print(f"X_train shape: {scaled_X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {scaled_X_test.shape}")
print(f"y_test shape: {y_test.shape}")
#########





#######################################################
# all data normalize
###########################################################

# Tüm verileri birle?tiriyoruz
all_data = pd.concat(monthly_dict_85to21.values(), axis=0)

# Tek bir scaler kullanarak normalize ediyoruz
scaler = MinMaxScaler()
normalized_all_data = pd.DataFrame(scaler.fit_transform(all_data), index=all_data.index, columns=all_data.columns)

# Normalized edilmi? veriyi her bir aya geri bölüyoruz
normalized_dataframes_eda = {}
start_idx = 0
for month, df in monthly_dict_85to21.items():
    end_idx = start_idx + len(df)
    normalized_dataframes_eda[month] = normalized_all_data.iloc[start_idx:end_idx]
    start_idx = end_idx



lookback = 36  # Modelin bakaca?? geçmi? ay say?s?
test_size = 24  # Tahmin edilecek 24 ay
batch_size = 256  # Bir seferde i?lenecek veri miktar?

X_train, y_train = [], []
X_test, y_test = [], []

# Her istasyon verisini e?itim ve test olarak ay?r?yoruz
for month, df in normalized_dataframes_eda.items():
    values = df.values  # DataFrame'i numpy array'e çeviriyoruz

    # E?itim verisi: lookback ve test_size d???nda kalan veriler
    num_samples = len(values) - lookback - test_size
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        X_train_batch = np.array([values[i:i + lookback] for i in range(start, end)])
        y_train_batch = np.array([values[i + lookback:i + lookback + test_size, 0] for i in range(start, end)])

        # X_train ve y_train listesine batch'leri ekleme
        X_train.append(X_train_batch)
        y_train.append(y_train_batch)

    # Test verisi: Son 12 ay ve test verisi
    X_test.append(values[-(lookback + test_size):-test_size])  # Son 12 ayl?k geçmi? veriyi al?yoruz
    y_test.append(values[-test_size:, 0])  # Sonraki 24 ay?n target verisi

# X_train ve y_train'i numpy array'lere çeviriyoruz (batch'ler halinde)
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
X_test = np.array(X_test)
y_test = np.array(y_test)

# E?itim ve test verilerinin ?ekillerini kontrol edelim
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


num_features = X_train.shape[2]  # Kaç özellik (sütun) var?
num_units = 30  # LSTM hücre say?s?

model = Sequential()
model.add(LSTM(num_units, input_shape=(lookback, num_features), return_sequences=True))
model.add(Dropout(0.2))  # Overfitting'i önlemek için
model.add(LSTM(num_units, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(24))  # 24 ayl?k tahmin

model.compile(optimizer='adam', loss='mse')

# Early stopping: Do?rulama kayb? 5 dönem boyunca iyile?mezse e?itim duracak
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


import time

# E?itim süresini ölçelim
start_time = time.time()
# Modeli e?itme
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

end_time = time.time()

print(f"Training took {end_time - start_time} seconds")

# Test verisi üzerinde tahmin yapma
predictions = model.predict(X_test)

# Tahmin sonuçlar? ve gerçek sonuçlar? inceleme
print("Tahmin edilen de?erler: ", predictions[0:5])
print("Gerçek test de?erleri: ", y_test[0:5])

# gerçek de?erleri ile görselle?tirme:

y_test_rescaled = []
predictions_rescaled = []

# Her bir istasyon için geri ölçeklendirme i?lemi
for i in range(len(X_test)):
    # Sadece 'Target' sütunu üzerinden geri ölçeklendirme yap?yoruz
    y_test_with_fake_columns = np.zeros((len(y_test[i]), X_train.shape[2]))
    y_test_with_fake_columns[:, -1] = y_test[i].reshape(-1)

    predictions_with_fake_columns = np.zeros((len(predictions[i]), X_train.shape[2]))
    predictions_with_fake_columns[:, -1] = predictions[i].reshape(-1)

    # Geri ölçeklendirme
    y_test_rescaled.append(scaler.inverse_transform(y_test_with_fake_columns)[:, -1])
    predictions_rescaled.append(scaler.inverse_transform(predictions_with_fake_columns)[:, -1])


# Örnek bir istasyon (station_index) için tahmin ve gerçek de?erlerin grafi?ini çizelim
station_index = 400

# Zaman aral??? (24 ay)
time_range = np.arange(1, len(y_test_rescaled[station_index]) + 1)

# Grafik olu?turma
plt.figure(figsize=(10, 6))
plt.plot(time_range, y_test_rescaled[station_index], label='Gerçek De?erler', marker='o')
plt.plot(time_range, predictions_rescaled[station_index], label='Tahmin Edilen De?erler', marker='x')
plt.title(f'?stasyon {station_index} için Gerçek ve Tahmin Edilen De?erler (Orijinal Ölçek)')
plt.xlabel('Ay')
plt.ylabel('Yer Alt? Su Yüksekli?i')
plt.legend()
plt.grid(True)
plt.show()

smape_value1 = smape(y_test_rescaled, predictions_rescaled)
print(f"SMAPE: {smape_value1}")

smape_value2 = smape(y_test, predictions)
print(f"SMAPE: {smape_value2}")