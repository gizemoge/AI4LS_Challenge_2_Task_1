# IMPORTS
import os
import pandas as pd
from datetime import datetime
from scipy.spatial import distance
import seaborn as sns
import itertools
import pandas as pd
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


# FUNCTIONS
def station_coordinates(input):
    """
    Creates a dataset consisting of measurement station IDs and their corresponding X and Y coordinates.

    Args:
        input: Directory of the measurement station CSV file.

    Returns:
        df: A DataFrame containing columns "x", "y", and "hzbnr01".
    """
    df = pd.read_csv(f"Ehyd/datasets_ehyd/{input}/messstellen_alle.csv", sep=";")
    output_df = df[["x", "y", "hzbnr01"]].copy()
    output_df['x'] = output_df['x'].astype(str).str.replace(',', '.').astype(float)
    output_df['y'] = output_df['y'].astype(str).str.replace(',', '.').astype(float)
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
                                    value = float(value_str.replace(',', '.'))
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
groundwater_folder_path = "Ehyd/datasets_ehyd/Groundwater/Grundwasserstand-Monatsmittel"
groundwater_dict, groundwater_coordinates = to_dataframe(groundwater_folder_path, groundwater_all_coordinates)

to_global(groundwater_dict, prefix="gw_")

# Groundwater Temperature
groundwater_temperature_folder_path = "Ehyd\datasets_ehyd\Groundwater\Grundwassertemperatur-Monatsmittel"
groundwater_temperature_dict, groundwater_temperature_coordinates = to_dataframe(groundwater_temperature_folder_path, groundwater_all_coordinates)

to_global(groundwater_temperature_dict, prefix="gwt_")

# Creating new dictionaries according to requested stations
points = pd.read_csv("Ehyd/datasets_ehyd/gw_test_empty.csv")
points_list = [col for col in points.columns[1:]]

filtered_groundwater_dict = filter_dataframes_by_points(groundwater_dict, points_list)
filtered_gw_coordinates = groundwater_coordinates[groundwater_coordinates['hzbnr01'].isin([int(i) for i in points_list])]

##################################### Precipitation
precipitation_coordinates = station_coordinates("Precipitation")

# Rain
rain_folder_path = "Ehyd/datasets_ehyd/Precipitation/N-Tagessummen"
rain_dict, rain_coordinates = to_dataframe(rain_folder_path, precipitation_coordinates)
rain_dict_monthly = to_monthly(rain_dict)
to_global(rain_dict_monthly, prefix="rain_")

# Snow
snow_folder_path = "Ehyd/datasets_ehyd/Precipitation/NS-Tagessummen"
snow_dict, snow_coordinates = to_dataframe(snow_folder_path, precipitation_coordinates)
snow_dict_monthly = to_monthly(snow_dict)
to_global(snow_dict_monthly, prefix="snow_")


##################################### Sources
sources_coordinates = station_coordinates("Sources")

# Flow Rate
source_flow_rate_path = "Ehyd/datasets_ehyd/Sources/Quellschüttung-Tagesmittel"
source_flow_rate_dict, source_flow_rate_coordinates = to_dataframe(source_flow_rate_path, sources_coordinates)
source_flow_rate_dict_monthly = to_monthly(source_flow_rate_dict)
to_global(source_flow_rate_dict_monthly, prefix="source_fr_")

# Conductivity
conductivity_folder_path = "Ehyd/datasets_ehyd/Sources/Quellleitfähigkeit-Tagesmittel"
conductivity_dict, conductivity_coordinates = to_dataframe(conductivity_folder_path, sources_coordinates)
conductivity_dict_monthly = to_monthly(conductivity_dict)
to_global(conductivity_dict_monthly, prefix="conductivity_")

# Source Temperature
source_temp_folder_path = "Ehyd/datasets_ehyd/Sources/Quellwassertemperatur-Tagesmittel"
source_temp_dict, source_temp_coordinates = to_dataframe(source_temp_folder_path, sources_coordinates)
source_temp_dict_monthly = to_monthly(source_temp_dict)
to_global(source_temp_dict_monthly, prefix="source_temp_")

##################################### Surface Water

surface_water_coordinates = station_coordinates("Surface_Water")

# Surface Water Level
surface_water_level_folder_path = "Ehyd/datasets_ehyd/Surface_Water/W-Tagesmittel"
surface_water_level_dict, surface_water_level_coordinates = to_dataframe(surface_water_level_folder_path, surface_water_coordinates)
surface_water_level_dict_monthly = to_monthly(surface_water_level_dict)
to_global(surface_water_level_dict, prefix="surface_water_level")

# Surface Water Temperature
surface_water_temp_folder_path = "Ehyd/datasets_ehyd/Surface_Water/WT-Monatsmittel"
river_temp_dict, river_temp_coordinates = to_dataframe(surface_water_temp_folder_path, surface_water_coordinates)
to_global(river_temp_dict, prefix="surface_water_temp")

# Sediment
sediment_folder_path = "Ehyd/datasets_ehyd/Surface_Water/Schwebstoff-Tagesfracht"
sediment_dict, sediment_coordinates = to_dataframe(sediment_folder_path, surface_water_coordinates)  # daily version
sediment_dict_monthly = to_monthly(sediment_dict)
to_global(sediment_dict_monthly, prefix="sediment_")

# Surface Water Flow Rate
surface_water_flow_rate_folder_path = "Ehyd/datasets_ehyd/Surface_Water/Q-Tagesmittel"
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
data = add_nearest_coordinates_column(rain_coordinates, 'nearest_rain', 20, df_to_merge=data)
data = add_nearest_coordinates_column(snow_coordinates, 'nearest_snow', 15, df_to_merge=data)
data = add_nearest_coordinates_column(source_flow_rate_coordinates, 'nearest_source_fr', 5, df_to_merge=data)
data = add_nearest_coordinates_column(conductivity_coordinates, 'nearest_conductivity', 5, df_to_merge=data)
data = add_nearest_coordinates_column(source_temp_coordinates, 'nearest_source_temp', 5, df_to_merge=data)
data = add_nearest_coordinates_column(surface_water_level_coordinates, 'nearest_owf_level', 20, df_to_merge=data)
data = add_nearest_coordinates_column(river_temp_coordinates, 'nearest_owf_temp', 10, df_to_merge=data)
data = add_nearest_coordinates_column(sediment_coordinates, 'nearest_sediment', 1, df_to_merge=data)
data = add_nearest_coordinates_column(surface_water_flow_rate_coordinates, 'nearest_owf_fr', 20, df_to_merge=data)
data.drop(["x", "y"], axis=1, inplace=True)


########################################################################################################################
# Investigating the dataframes
########################################################################################################################
def is_monoton(dict):
    for df_name, value in dict.items():
        if value['Date'].is_monotonic_increasing == False:
            print(f"{df_name} monoton art?? göstermiyor. Bu konuda bir aksiyon al?nmal?")

is_monoton(filtered_groundwater_dict)
is_monoton(groundwater_temperature_dict)
is_monoton(rain_dict)
is_monoton(snow_dict)
is_monoton(source_flow_rate_dict)
is_monoton(conductivity_dict)
is_monoton(source_temp_dict)
is_monoton(surface_water_level_dict)
is_monoton(river_temp_dict)
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


for df_name, df in rain_dict.items():
    nan_rows = df[df.isnull().any(axis=1)]
    print(f"DataFrame: {df_name}")
    print(f"Toplam NaN say?s?: {df.isnull().sum()}")
    print(nan_rows)


########################################################################################################################
# Imputing Missing Values with SARIMA()
########################################################################################################################
deneme = rain_126011.copy()

# SARIMA
# Parametre kombinasyonlar?n? olu?turma
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

best_aic = float("inf")
best_param = None
best_seasonal_param = None

# En iyi SARIMA parametrelerini bulmak için Grid Search
for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            model = SARIMAX(deneme['Values'],
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

print(f"En iyi parametreler: {best_param} ve {best_seasonal_param} ile AIC: {best_aic}")

# SARIMA modelini olu?turma ve tahmin yapma
sarima_model = SARIMAX(deneme['Values'],
                       order=best_param,
                       seasonal_order=best_seasonal_param)
sarima_result = sarima_model.fit(disp=False)

# Eksik de?erleri doldurma
deneme['values_filled'] = sarima_result.predict(start=deneme.index[0], end=deneme.index[-1])

deneme.drop(deneme.index[0:15], inplace=True)


# Sonuçlar? görselle?tirme
plt.figure(figsize=(12, 6))
plt.plot(deneme.index, deneme['Values'], label='Original Values', linestyle='--', color='blue')
plt.plot(deneme.index, deneme['values_filled'], label='Filled Values', linestyle='-', color='red')
plt.title('Original and Filled Values, lag=-1, SARIMA')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


