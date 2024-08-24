# IMPORTS
import os
import pandas as pd
from datetime import datetime
from scipy.spatial import distance
import seaborn as sns
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
    Ölçüm istasyonu ID'si ve X ve Y koordinatlar?ndan olu?an bir veri seti yarat?r.

    Args:
        input: ölçüm istasyonu CSV directory'si

    Returns:
        df: df["x", "y", "hzbnr01"]
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

                        df_name = f"{filename[-10:-4]}"
                        dataframes_dict[df_name] = df

                        # Convert keys to integers
                        int_keys = [int(key) for key in dataframes_dict.keys() if key.isdigit()]
                        coordinates = tip_coordinates[tip_coordinates['hzbnr01'].isin(int_keys)]

        except Exception:
            continue

    return dataframes_dict, coordinates

def to_global(dataframes_dict, prefix=''):
    for name, dataframe in dataframes_dict.items():
        globals()[f"{prefix}{name}"] = dataframe

def filter_dataframes_by_points(dataframes_dict, points_list):
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
filtered_groundwater_temp_dict = filter_dataframes_by_points(groundwater_temperature_dict, points_list)


##################################### Precipitation
precipitation_coordinates = station_coordinates("Precipitation")

# Rain
rain_folder_path = "Ehyd/datasets_ehyd/Precipitation/N-Tagessummen"
rain_dict, rain_coordinates = to_dataframe(rain_folder_path, precipitation_coordinates)

to_global(rain_dict, prefix="rain_")

# Snow
snow_folder_path = "Ehyd/datasets_ehyd/Precipitation/NS-Tagessummen"
snow_dict, snow_coordinates = to_dataframe(snow_folder_path, precipitation_coordinates)

to_global(snow_dict, prefix="snow_")


##################################### Sources
sources_coordinates = station_coordinates("Sources")

# Flow Rate
source_flow_rate_path = "Ehyd/datasets_ehyd/Sources/Quellschüttung-Tagesmittel"
source_flow_rate_dict, source_flow_rate_coordinates = to_dataframe(source_flow_rate_path, sources_coordinates)

to_global(source_flow_rate_dict, prefix="source_fr_")

# Conductivity
conductivity_folder_path = "Ehyd/datasets_ehyd/Sources/Quellleitfähigkeit-Tagesmittel"
conductivity_dict, conductivity_coordinates = to_dataframe(conductivity_folder_path, sources_coordinates)

to_global(conductivity_dict, prefix="conductivity_")

# Source Temperature
source_temp_folder_path = "Ehyd/datasets_ehyd/Sources/Quellwassertemperatur-Tagesmittel"
source_temp_dict, source_temp_coordinates = to_dataframe(source_temp_folder_path, sources_coordinates)

to_global(source_temp_dict, prefix="source_temp_")


##################################### Surface Water

surface_water_coordinates = station_coordinates("Surface_Water")

# River Water Level
river_level_folder_path = "Ehyd/datasets_ehyd/Surface_Water/W-Tagesmittel"
river_level_dict, river_level_coordinates = to_dataframe(river_level_folder_path, surface_water_coordinates)

to_global(river_level_dict, prefix="river_level")

# River Water Temperature
river_temp_folder_path = "Ehyd/datasets_ehyd/Surface_Water/WT-Monatsmittel"
river_temp_dict, river_temp_coordinates = to_dataframe(river_temp_folder_path, surface_water_coordinates)

to_global(river_temp_dict, prefix="river_temp")

# Sediment
sediment_folder_path = "Ehyd/datasets_ehyd/Surface_Water/Schwebstoff-Tagesfracht"
sediment_dict, sediment_coordinates = to_dataframe(sediment_folder_path, surface_water_coordinates)

to_global(sediment_dict, prefix="sediment_")

# River Water Flow Rate
river_flow_rate_folder_path = "Ehyd/datasets_ehyd/Surface_Water/Q-Tagesmittel"
river_flow_rate_dict, river_flow_rate_coordinates = to_dataframe(river_flow_rate_folder_path, surface_water_coordinates)

to_global(river_flow_rate_dict, prefix="river_fr_")


########################################################################################################################
# Her bir gw_coordinates noktas?n?n en yak?n 10 rain_coordinates noktas?n? bul
def calculate_distance(coord1, coord2):
    return distance.euclidean(coord1, coord2)


def find_nearest_rain_coordinates(gw_row, rain_df, k=20):
    distances = rain_df.apply(lambda row: calculate_distance(
        (gw_row['x'], gw_row['y']),
        (row['x'], row['y'])
    ), axis=1)
    nearest_indices = distances.nsmallest(k).index
    return rain_df.loc[nearest_indices]


results = []
for _, gw_row in filtered_gw_coordinates.iterrows():
    nearest_rains = find_nearest_rain_coordinates(gw_row, rain_coordinates)
    nearest_owf = # TODO
    nearest_qu = # TODO
    results.append({
        'gw_hzbnr01': gw_row['hzbnr01'],
        'nearest_rain_hzbnr01': nearest_rains['hzbnr01'].tolist()

    })

# Sonu?lar? bir veri ?er?evesine d?n??t?r
results_df = pd.DataFrame(results)



# biri nokta bazl? bakmak: df_gw_379313
to_global(filtered_gw_dataframes_dict)
df_gw_379313 = filtered_gw_dataframes_dict["df_gw_379313"]  # istenilen noktalardan birisi
df_gw_379313.head()

df_gw_379313["Date"].min()  # (1980, 12, 1)
df_gw_379313["Date"].max()  # (2022, 1, 1)

df_gw_379313.isnull().sum()  # 54 null var


# adding rain
to_global(rain_dataframes_dict)
df_379313_rain = results_df[results_df["gw_hzbnr01"] == "df_gw_379313"]["nearest_rain_hzbnr01"]
df_379313_rain_list = df_379313_rain.iloc[0]



for df_name in df_379313_rain_list:
    df = globals().get(df_name)
    print(f"{df_name} null values:\n{df.isnull().sum()}\n")



# rain'i ayl?k yapmak
# concat'lamadan ?nce ya?muru g?nl?kten ayl??a ?evirmek gerek
for df_name in df_379313_rain_list:
    df = globals()[df_name]  # DataFrame'i ismiyle global de?i?kenlerden al
    df['Date'] = pd.to_datetime(df['Date'])  # Date s?tununu datetime format?na ?evir
    df.set_index('Date', inplace=True)  # Date s?tununu index olarak ayarla
    df_monthly = df.resample('ME').mean().reset_index()  # Ayl?k ortalamay? hesapla ve reset index
    globals()[df_name] = df_monthly


# min max tarihlerine bakmak
for df_name in df_379313_rain_list:
    df = globals().get(df_name)
    date = "Date"
    print(f"{df_name} min values:\n{df[date].min()}\n")
    print(f"{df_name} max values:\n{df[date].max()}\n")
    print("\n")

# df_115352 min values:
# 2011-01-31 00:00:00
# df_115352 max values:
# 2022-01-31 00:00:00

df_379313["Date"].min() # ('1980-12-01 00:00:00')
df_379313["Date"].max() # ('2022-01-01 00:00:00')
#########################################################################################################################3

# Tarihler aras?nda filtreleme yap
df_379313["Date"] = pd.to_datetime(df_379313["Date"])
# df_379313 = df_379313[(df_379313["Date"] >= "1980-12-01") & (df_379313["Date"] <= "2021-12-01")]



def set_day_to_first(df):
    df['Date'] = df['Date'].apply(lambda x: x.replace(day=1))
    return df

for df_name in df_379313_rain_list:  # bu, a?a?daki fonksiyon ile birle?tirilebilir
    df = globals().get(df_name)
    df = set_day_to_first(df)
    df = df[(df["Date"] >= "1980-12-01") & (df["Date"] <= "2021-12-01")]
    print(df.head())


def fill_missing_dates_nan(df, start_date="1980-12-01", end_date="2021-12-01"):
    # T?m tarih aral???n? olu?tur
    all_dates = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Veri ?er?evesini t?m tarihlerle yeniden indeksleyin ve eksik de?erleri NaN yap?n
    df = df.set_index('Date').reindex(all_dates).reset_index()
    df.columns = ['Date'] + list(df.columns[1:])  # 'Date' kolonunu tekrar adland?r?n

    return df


for df_name in df_379313_rain_list:
    df = globals().get(df_name)
    if df is not None:
        df = fill_missing_dates_nan(df)
        globals()[df_name] = df
        print(df.head())




merge_list = [df_379313]
for df_name in df_379313_rain_list:
    df = globals().get(df_name)
    merge_list.append(df)


# merge
merged_df = merge_list[0]

# Listeyi gezerek t?m DataFrame'leri birle?tiriyoruz
for i, df in enumerate(merge_list[1:], start=1):
    # Her yeni DataFrame'den gelen kolonlara _i ekliyoruz
    suffixes = (None, f'_df{i}')
    merged_df = pd.merge(merged_df, df, on='Date', how='inner', suffixes=suffixes)



merged_df.columns = ['Date', 'df_379313', 'df_100933', 'df_100925', 'df_100404', 'df_100941', 'df_100412', 'df_100503', 'df_100883', 'df_115352', 'df_100370', 'df_100529']
merged_df.head()

# Lagged merged df

# Lag de?erini eklemek istedi?iniz kolonlar
cols_to_lag = ['df_100933', 'df_100925', 'df_100404', 'df_100941', 'df_100412', 'df_100503', 'df_100883', 'df_115352', 'df_100370', 'df_100529']

# Yeni bir veri ?er?evesi olu?tur
lagged_merged_df = merged_df.copy()

# Lag kolonlar?n? olu?tur
for col in cols_to_lag:
    lagged_merged_df[f'{col}_lag1'] = merged_df[col].shift(1)
    lagged_merged_df[f'{col}_lag2'] = merged_df[col].shift(2)

# Kolonlar? s?ralamak i?in ?nce lag1'ler ve sonra lag2'ler
# ?nce t?m kolonlar? listeleyin
lag1_cols = [f'{col}_lag1' for col in cols_to_lag]
lag2_cols = [f'{col}_lag2' for col in cols_to_lag]
merged_df_cols = merged_df.columns.to_list()

# Kolon s?ralamas?
new_order = merged_df_cols  + lag1_cols + lag2_cols

# Veri ?er?evesini yeni kolon s?ralamas?na g?re d?zenle
lagged_merged_df = lagged_merged_df[new_order]

# Sonu?lar? g?r?nt?leme
print(lagged_merged_df.head())

len(lagged_merged_df.columns)

# Heatmap
data = lagged_merged_df.iloc[:, -31:]
plt.figure(figsize=(20, 20))
dataplot = sns.heatmap(data.corr(), cmap="YlGnBu", annot=True, annot_kws={'size': 8})

# Grafi?i kaydetmek
plt.savefig("Ehyd/datasets_ehyd/heatmap_80s_with_nan_n_lags.png", bbox_inches='tight')

# Grafi?i g?stermek
plt.show()


merged_df.shape  # (493, 12)
merged_df.isnull().sum()
# merged_df.isnull().sum()
# Out[85]:
# Date           0
# df_379313     53
# df_100933    128
# df_100925    109
# df_100404      0
# df_100941    127
# df_100412      0
# df_100503      0
# df_100883    109
# df_115352    361
# df_100370      0
# df_100529      0
# dtype: int64
