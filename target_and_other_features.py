# bu dosyada targeta rain ve snow verilerimni eklemeyi deniyorum.

import pandas as pd
import os

# matches.csv dosyas?n? oku
matches_df = pd.read_csv('datasets/matches.csv', sep=";")
matches_df.head()


def process_target_csv(file_path, matches_df, rain_directory, snow_directory):
    # Target CSV dosyas?n? oku
    target_df = pd.read_csv(file_path, sep=";")
    target_hzbnr01 = os.path.basename(file_path).replace(".csv", "")

    # matches.csv dosyas?ndaki ilgili sat?r? bul
    match_row = matches_df[matches_df['hzbnr01'] == int(target_hzbnr01)]

    if match_row.empty:
        print(f"No match found for {target_hzbnr01}")
        return

    # En yak?n kom?u kodlar?n? ve a??rl?klar?n? al
    rain_neighbors = match_row.filter(like='nearest_nlv_rain_').values.flatten()
    rain_weights = match_row.filter(like='nlv_rain_weight_').values.flatten()
    snow_neighbors = match_row.filter(like='nearest_nlv_snow_').values.flatten()
    snow_weights = match_row.filter(like='nlv_snow_weight_').values.flatten()

    # 'date' sütununu datetime format?na çevir
    target_df['Date'] = pd.to_datetime(target_df['Date'], format='%Y-%m-%d')

    # Ya?mur ve kar verilerini hesapla
    rain_value_sum = pd.Series(0, index=target_df.index, dtype=float)
    snow_value_sum = pd.Series(0, index=target_df.index, dtype=float)

    for neighbor, weight in zip(rain_neighbors, rain_weights):
        neighbor_file = os.path.join(rain_directory, f"{int(neighbor)}.csv")
        if os.path.exists(neighbor_file):
            neighbor_df = pd.read_csv(neighbor_file, sep=";")
            neighbor_df['Date'] = pd.to_datetime(neighbor_df['Date'], format='%Y-%m-%d')
            for idx, row in target_df.iterrows():
                matching_row = neighbor_df[(neighbor_df['Date'].dt.year == row['Date'].year) &
                                           (neighbor_df['Date'].dt.month == row['Date'].month)]
                if not matching_row.empty:
                    rain_value_sum[idx] += matching_row.iloc[0, 1] * weight

    for neighbor, weight in zip(snow_neighbors, snow_weights):
        neighbor_file = os.path.join(snow_directory, f"{int(neighbor)}.csv")
        if os.path.exists(neighbor_file):
            neighbor_df = pd.read_csv(neighbor_file, sep=";")
            neighbor_df['Date'] = pd.to_datetime(neighbor_df['Date'], format='%Y-%m-%d')
            for idx, row in target_df.iterrows():
                matching_row = neighbor_df[(neighbor_df['Date'].dt.year == row['Date'].year) &
                                           (neighbor_df['Date'].dt.month == row['Date'].month)]
                if not matching_row.empty:
                    snow_value_sum[idx] += matching_row.iloc[0, 1] * weight

    # Yeni sütunlar ekle
    target_df['rain'] = rain_value_sum
    target_df['snow'] = snow_value_sum

    # Yeni dosyay? kaydet
    output_file_path = file_path.replace("processed", "processed_new")
    target_df.to_csv(output_file_path, index=False)
    print(f"Processed {file_path} -> {output_file_path}")


# Klasör yollar?
processed_directory = 'datasets/processed'
rain_directory = 'datasets/processed_rain'
snow_directory = 'datasets/processed_snow'

# Yeni klasör olu?tur
if not os.path.exists('datasets/processed_new'):
    os.makedirs('datasets/processed_new')

# processed klasöründeki tüm CSV dosyalar?n? i?le
for filename in os.listdir(processed_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(processed_directory, filename)
        process_target_csv(file_path, matches_df, rain_directory, snow_directory)
