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

# functions
def create_dataframes_from_csv(folder_path):

    dataframes_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)

            with open(filepath, 'r', encoding='latin1') as file:
                lines = file.readlines()

                start_idx = next(i for i, line in enumerate(lines) if line.startswith("Werte:")) + 1
                data_lines = lines[start_idx:]

                data = []
                for line in data_lines:
                    if line.strip():  # Bo? sat?rlar? geç
                        date_str, value_str = line.split(';')[:2]
                        date = datetime.strptime(date_str.strip(), "%d.%m.%Y %H:%M:%S").date()
                        value = value_str.strip().replace('Lücke', 'NaN')  # 'Lücke' olanlar? NaN yap
                        data.append([date, float(value.replace(',', '.'))])

                df = pd.DataFrame(data, columns=['Date', 'Values'])

                df_name = f"df_{filename[-10:-4]}"

                # Sözlü?e ekle
                dataframes_dict[df_name] = df

    return dataframes_dict

def to_global(dataframes_dict):
    for name, dataframe in dataframes_dict.items():
        globals()[name] = dataframe


# groundwater
# level
gw_folder_path = "Ehyd/datasets_ehyd/Groundwater/Grundwasserstand-Monatsmittel"
gw_dataframes_dict = create_dataframes_from_csv(gw_folder_path)

to_global(gw_dataframes_dict)
df_300012.head()
df_300012.columns
# todo üç kolon olu?uyor mu


# temperature
gwt_folder_path = "Ehyd/datasets_ehyd/Groundwater/Grundwassertemperatur-Monatsmittel"
gwt_dataframes_dict = create_dataframes_from_csv(gwt_folder_path)
to_global(gw_dataframes_dict)
gwt_dataframes_dict


points = pd.read_csv("Ehyd/datasets_ehyd/gw_test_empty.csv")
points_list = ['df_' + col for col in points.columns[1:]]

for df_name in points_list:
    df = globals().get(df_name)
    date = "Date"
    print(f"{df_name} min values:\n{df[date].min()}\n")
    print(f"{df_name} max values:\n{df[date].max()}\n")
    print("\n")



# 487'ye filtreleme
def filter_dataframes_by_points(dataframes_dict, points_list):
    filtered_dict = {name: df for name, df in dataframes_dict.items() if name in points_list}
    return filtered_dict
filtered_gw_dataframes_dict = filter_dataframes_by_points(gw_dataframes_dict, points_list)

len(filtered_gw_dataframes_dict)

# coordinates
gw_coordinates = pd.read_csv("Ehyd/datasets_ehyd/Groundwater/messstellen_alle.csv", sep=";")
gw_coordinates.shape  # (3792, 5)

gw_coordinates["hzbnr01"] = gw_coordinates["hzbnr01"].apply(lambda x: f"df_{x}")

gw_coordinates['x'] = gw_coordinates['x'].astype(str).str.replace(',', '.').astype(float)
gw_coordinates['y'] = gw_coordinates['y'].astype(str).str.replace(',', '.').astype(float)

# 487'ye indiriyorum filtered diye at?yorum, çünkü di?er veirlmeyen yak?n noktalra? da kullanmak istiyorum
filtered_gw_coordinates = gw_coordinates[gw_coordinates['hzbnr01'].isin(points_list)]



# rain
rain_folder_path = "Ehyd/datasets_ehyd/Precipitation/N-Tagessummen"
rain_dataframes_dict = create_dataframes_from_csv(rain_folder_path)

# precipitation coordinates (rain + snow)
precipitation_coordinates = pd.read_csv("Ehyd/datasets_ehyd/Precipitation/messstellen_alle.csv", sep=";")
precipitation_coordinates.head()

precipitation_coordinates["hzbnr01"] = precipitation_coordinates["hzbnr01"].apply(lambda x: f"df_{x}")

# rain coordinates
rain_keys = list(rain_dataframes_dict.keys())
rain_coordinates = precipitation_coordinates[precipitation_coordinates['hzbnr01'].isin(rain_keys)].copy()
rain_coordinates.shape

rain_keys # bu rain rain'in dataframe'lerinin listesi

rain_coordinates['x'] = rain_coordinates['x'].astype(str).str.replace(',', '.').astype(float)
rain_coordinates['y'] = rain_coordinates['y'].astype(str).str.replace(',', '.').astype(float)





# Her bir gw_coordinates noktas?n?n en yak?n 10 rain_coordinates noktas?n? bul
def calculate_distance(coord1, coord2):
    return distance.euclidean(coord1, coord2)


def find_nearest_rain_coordinates(gw_row, rain_df, k=10):
    distances = rain_df.apply(lambda row: calculate_distance(
        (gw_row['x'], gw_row['y']),
        (row['x'], row['y'])
    ), axis=1)
    nearest_indices = distances.nsmallest(k).index
    return rain_df.loc[nearest_indices]


results = []
for _, gw_row in filtered_gw_coordinates.iterrows():
    nearest_rains = find_nearest_rain_coordinates(gw_row, rain_coordinates)
    results.append({
        'gw_hzbnr01': gw_row['hzbnr01'],
        'nearest_rain_hzbnr01': nearest_rains['hzbnr01'].tolist()
    })

# Sonuçlar? bir veri çerçevesine dönü?tür
results_df = pd.DataFrame(results)

print(results_df.head())

results_df.shape  # 487



# biri nokta bazl? bakmak:
to_global(filtered_gw_dataframes_dict)
df_379313 = filtered_gw_dataframes_dict["df_379313"]  # istenilen noktalardan birisi
df_379313.head()

df_379313["Date"].min()  # (1980, 12, 1)
df_379313["Date"].max()  # (2022, 1, 1)

df_379313.isnull().sum()  # 54 null var


# adding rain
to_global(rain_dataframes_dict)
df_379313_rain = results_df[results_df["gw_hzbnr01"] == "df_379313"]["nearest_rain_hzbnr01"]
df_379313_rain_list = df_379313_rain.iloc[0]
# [df_100933, df_100925, df_100404, df_100941, df_100412, df_100503, df_100883, df_115352, df_100370, df_100529]


for df_name in df_379313_rain_list:
    df = globals().get(df_name)
    print(f"{df_name} null values:\n{df.isnull().sum()}\n")



# rain'i ayl?k yapmak
# concat'lamadan önce ya?muru günlükten ayl??a çevirmek gerek
for df_name in df_379313_rain_list:
    df = globals()[df_name]  # DataFrame'i ismiyle global de?i?kenlerden al
    df['Date'] = pd.to_datetime(df['Date'])  # Date sütununu datetime format?na çevir
    df.set_index('Date', inplace=True)  # Date sütununu index olarak ayarla
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
    # Tüm tarih aral???n? olu?tur
    all_dates = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Veri çerçevesini tüm tarihlerle yeniden indeksleyin ve eksik de?erleri NaN yap?n
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

# Listeyi gezerek tüm DataFrame'leri birle?tiriyoruz
for i, df in enumerate(merge_list[1:], start=1):
    # Her yeni DataFrame'den gelen kolonlara _i ekliyoruz
    suffixes = (None, f'_df{i}')
    merged_df = pd.merge(merged_df, df, on='Date', how='inner', suffixes=suffixes)



merged_df.columns = ['Date', 'df_379313', 'df_100933', 'df_100925', 'df_100404', 'df_100941', 'df_100412', 'df_100503', 'df_100883', 'df_115352', 'df_100370', 'df_100529']
merged_df.head()

# Lagged merged df

# Lag de?erini eklemek istedi?iniz kolonlar
cols_to_lag = ['df_100933', 'df_100925', 'df_100404', 'df_100941', 'df_100412', 'df_100503', 'df_100883', 'df_115352', 'df_100370', 'df_100529']

# Yeni bir veri çerçevesi olu?tur
lagged_merged_df = merged_df.copy()

# Lag kolonlar?n? olu?tur
for col in cols_to_lag:
    lagged_merged_df[f'{col}_lag1'] = merged_df[col].shift(1)
    lagged_merged_df[f'{col}_lag2'] = merged_df[col].shift(2)

# Kolonlar? s?ralamak için önce lag1'ler ve sonra lag2'ler
# Önce tüm kolonlar? listeleyin
lag1_cols = [f'{col}_lag1' for col in cols_to_lag]
lag2_cols = [f'{col}_lag2' for col in cols_to_lag]
merged_df_cols = merged_df.columns.to_list()

# Kolon s?ralamas?
new_order = merged_df_cols  + lag1_cols + lag2_cols

# Veri çerçevesini yeni kolon s?ralamas?na göre düzenle
lagged_merged_df = lagged_merged_df[new_order]

# Sonuçlar? görüntüleme
print(lagged_merged_df.head())

len(lagged_merged_df.columns)

# Heatmap
data = lagged_merged_df.iloc[:, -31:]
plt.figure(figsize=(20, 20))
dataplot = sns.heatmap(data.corr(), cmap="YlGnBu", annot=True, annot_kws={'size': 8})

# Grafi?i kaydetmek
plt.savefig("Ehyd/datasets_ehyd/heatmap_80s_with_nan_n_lags.png", bbox_inches='tight')

# Grafi?i göstermek
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
