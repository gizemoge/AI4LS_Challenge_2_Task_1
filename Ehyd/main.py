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

def to_monthly(df_dict):
    for df_name, df_value in df_dict.items():
        df_value['Date'] = pd.to_datetime(df_value['Date'])
        df_value.set_index('Date', inplace=True)
        df_dict[df_name] = df_value.resample('M').mean()  # Do?rudan sözlü?ü güncelle
    return df_dict

def date_to_index(df_dict):
    for df_name, df_value in df_dict.items():
        df_value['Date'] = pd.to_datetime(df_value['Date'])
        df_value.set_index('Date', inplace=True)
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

##################################### Groundwater
groundwater_all_coordinates = station_coordinates("Groundwater")

# Groundwater Level
groundwater_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Groundwater", "Grundwasserstand-Monatsmittel")
groundwater_dict, groundwater_coordinates = to_dataframe(groundwater_folder_path, groundwater_all_coordinates)
groundwater_dict = date_to_index(groundwater_dict)
to_global(groundwater_dict, prefix="gw_")

# Groundwater Temperature
groundwater_temperature_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Groundwater", "Grundwassertemperatur-Monatsmittel")
groundwater_temperature_dict, groundwater_temperature_coordinates = to_dataframe(groundwater_temperature_folder_path, groundwater_all_coordinates)
groundwater_temperature_dict = date_to_index(groundwater_temperature_dict)
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
rain_dict_monthly = to_monthly(rain_dict)
to_global(rain_dict_monthly, prefix="rain_")

# Snow
snow_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Precipitation", "NS-Tagessummen")
snow_dict, snow_coordinates = to_dataframe(snow_folder_path, precipitation_coordinates)
snow_dict_monthly = to_monthly(snow_dict)
to_global(snow_dict_monthly, prefix="snow_")


##################################### Sources
sources_coordinates = station_coordinates("Sources")

# Flow Rate
source_flow_rate_path = os.path.join("Ehyd", "datasets_ehyd", "Sources", "Quellschüttung-Tagesmittel")
source_flow_rate_dict, source_flow_rate_coordinates = to_dataframe(source_flow_rate_path, sources_coordinates)
source_flow_rate_dict_monthly = to_monthly(source_flow_rate_dict)
to_global(source_flow_rate_dict_monthly, prefix="source_fr_")

# Conductivity
conductivity_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Sources", "Quellleitfähigkeit-Tagesmittel")
conductivity_dict, conductivity_coordinates = to_dataframe(conductivity_folder_path, sources_coordinates)
conductivity_dict_monthly = to_monthly(conductivity_dict)
to_global(conductivity_dict_monthly, prefix="conductivity_")

# Source Temperature
source_temp_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Sources", "Quellwassertemperatur-Tagesmittel")
source_temp_dict, source_temp_coordinates = to_dataframe(source_temp_folder_path, sources_coordinates)
source_temp_dict_monthly = to_monthly(source_temp_dict)
to_global(source_temp_dict_monthly, prefix="source_temp_")

##################################### Surface Water

surface_water_coordinates = station_coordinates("Surface_Water")

# Surface Water Level
surface_water_level_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", "W-Tagesmittel")
surface_water_level_dict, surface_water_level_coordinates = to_dataframe(surface_water_level_folder_path, surface_water_coordinates)
surface_water_level_dict_monthly = to_monthly(surface_water_level_dict)
to_global(surface_water_level_dict, prefix="surface_water_level")

# Surface Water Temperature
surface_water_temp_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", "WT-Monatsmittel")
surface_water_temp_dict, surface_water_temp_coordinates = to_dataframe(surface_water_temp_folder_path, surface_water_coordinates)
surface_water_temp_dict = date_to_index(surface_water_temp_dict)

to_global(surface_water_temp_dict, prefix="surface_water_temp")

# Sediment
sediment_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", "Schwebstoff-Tagesfracht")
sediment_dict, sediment_coordinates = to_dataframe(sediment_folder_path, surface_water_coordinates)  # daily version
sediment_dict_monthly = to_monthly(sediment_dict)
to_global(sediment_dict_monthly, prefix="sediment_")

# Surface Water Flow Rate
surface_water_flow_rate_folder_path = os.path.join("Ehyd", "datasets_ehyd", "Surface_Water", "Q-Tagesmittel")
surface_water_flow_rate_dict, surface_water_flow_rate_coordinates = to_dataframe(surface_water_flow_rate_folder_path, surface_water_coordinates)
surface_water_flow_rate_dict_monthly = to_monthly(surface_water_flow_rate_dict)
to_global(surface_water_flow_rate_dict, prefix="surface_water_fr_")

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




# yer alt? suyu s?cakl?k tur?usu için
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




########################################################################################################################
# Investigating the dataframes
########################################################################################################################
def is_monoton(dict):
    for df_name, value in dict.items():
        if value.index.name != 'Date':
            if not value['Date'].is_monotonic_increasing:
                print(f"{df_name} monoton art?? göstermiyor. Bu konuda bir aksiyon al?nmal?")
            if 'Date' not in value.columns:
                print(f"---- {dict}{df_name} index veya sütun olarak 'Date' mevcut de?il.")
                continue
            continue
        if not value.index.is_monotonic_increasing:
            print(f"{df_name} monoton art?? göstermiyor. Bu konuda bir aksiyon al?nmal?")

is_monoton(filtered_groundwater_dict)
is_monoton(groundwater_temperature_dict)
is_monoton(rain_dict)
is_monoton(snow_dict)
is_monoton(source_flow_rate_dict)
is_monoton(conductivity_dict)
is_monoton(source_temp_dict)
is_monoton(surface_water_level_dict)
is_monoton(surface_water_temp_dict)
is_monoton(sediment_dict)
is_monoton(surface_water_flow_rate_dict)


def plot_row_count_distribution(df_dict):
    """
    Bu fonksiyon, verilen sözlükteki DataFrame'lerin sat?r say?lar?n?n da??l?m?n? histogram olarak çizer.

    Parametre:
    df_dict (dict): Anahtarlar?n string, de?erlerin ise pandas DataFrame oldu?u bir sözlük.
    """
    # DataFrame'lerin sat?r say?lar?n? hesaplay?n
    row_counts = [df.shape[0] for df in df_dict.values()]

    # Sat?r say?lar?n?n da??l?m?n? histogram olarak çizin
    plt.hist(row_counts, bins=10, edgecolor='black')
    plt.xlabel('Sat?r Say?s?')
    plt.ylabel('Frekans')
    plt.title('DataFrame Sat?r Say?lar?n?n Da??l?m?')
    plt.show()

plot_row_count_distribution(filtered_groundwater_dict)


shapes = []
dates = []

for df_name, value in filtered_groundwater_dict.items():
    shapes.append(value.shape[0])
    dates.append(value["Date"].min())

print(f"min: {min(shapes)} ay say?s?")
print(f"max: {max(shapes)} ay say?s?")
max(dates)  # max, min y?l 2001


for df_name, df in sediment_dict.items():
    nan_rows = df[df.isnull().any(axis=1)]
    print(f"DataFrame: {df_name}")
    print(f"Toplam NaN say?s?: {df.isnull().sum()}")
    print(nan_rows)


########################################################################################################################
# Imputing Missing Values with SARIMA()
########################################################################################################################

def fill_missing_values_with_sarima(df_dict):
    """
    Verilen DataFrame'lerdeki eksik de?erleri SARIMA modeli kullanarak doldurur.

    Args:
    df_dict (dict): Anahtarlar? DataFrame adlar?, de?erleri DataFrame'ler olan bir sözlük.

    Returns:
    dict: Eksik de?erleri doldurulmu? DataFrame'leri içeren bir sözlük.
    """
    filled_dfs = {}

    for key, df in df_dict.items():
        # SARIMA model parametrelerini belirleme
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

        best_aic = np.float32("inf")
        best_param = None
        best_seasonal_param = None

        # Grid Search ile en iyi SARIMA parametrelerini bulma
        for param in pdq:
            for seasonal_param in seasonal_pdq:
                try:
                    model = SARIMAX(df['Values'],
                                    order=param,
                                    seasonal_order=seasonal_param,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    results = model.fit(disp=False)
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_param = param
                        best_seasonal_param = seasonal_param
                except:
                    continue

        # En iyi parametrelerle SARIMA modelini olu?turma
        sarima_model = SARIMAX(df['Values'],
                               order=best_param,
                               seasonal_order=best_seasonal_param)
        sarima_result = sarima_model.fit(disp=False)

        # Eksik de?erleri doldurma
        df['Values_Filled'] = sarima_result.predict(start=df.index[0], end=df.index[-1])
        df['final_values'] = df['Values'].combine_first(df['Values_Filled'])
        df = df["final_values"]
        filled_dfs[key] = df

    return filled_dfs


# Fonksiyonun kullan?m?
filled_sediment_dict_monthly = fill_missing_values_with_sarima(sediment_dict_monthly)
filled_surface_water_level_dict_monthly = fill_missing_values_with_sarima(surface_water_level_dict_monthly)
filled_surface_water_flow_rate_dict_monthly = fill_missing_values_with_sarima(surface_water_flow_rate_dict_monthly)
filled_surface_water_temp_dict = fill_missing_values_with_sarima(surface_water_temp_dict)
# bunda bir uyar? verdi uyar?n?n çözümü: df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='MS')
# biz frekans belirtmedi?imiz için ayl?k oldu?unu tahmin ediyormu?
filled_groundwater_dict = fill_missing_values_with_sarima(filtered_groundwater_dict)
filled_snow_dict_monthly = fill_missing_values_with_sarima(snow_dict_monthly)
filled_data_gw_temp_dict = fill_missing_values_with_sarima(data_gw_temp_dict)
filled_conductivity_dict = fill_missing_values_with_sarima(conductivity_dict_monthly)
filled_source_flow_rate_dict = fill_missing_values_with_sarima(source_flow_rate_dict_monthly)
filled_source_temp_dict = fill_missing_values_with_sarima(source_temp_dict_monthly)
filled_rain_dict = fill_missing_values_with_sarima(rain_dict_monthly)




# write pickle
with open('filled_sediment_dict_monthly.pkl', 'wb') as f:
    pickle.dump(filled_sediment_dict_monthly, f)

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


################ 720 dataframe
list_of_dfs = []

# 720 ay için döngü
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
with open('list_of_dfs.pkld', 'rb') as f:
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

