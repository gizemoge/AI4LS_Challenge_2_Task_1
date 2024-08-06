# ?imdi nlv kodlar? hem ya?mur hem de kar verisini kaps?yor ama ben ikisiniz ayn? de?erlerndirilemeyece?ini dü?ünüyorum o yüzden burda
# onlar? ay?rmay? deniyorum

import os
import pandas as pd


def find_matching_files(hzbnr01_codes, directory):
    matching_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Dosya ad?ndan uzant?y? kald?rarak sadece hzbnr01 kodunu al
            file_hzbnr01 = filename.replace(".csv", "")
            if file_hzbnr01 in hzbnr01_codes:
                matching_files.append(file_hzbnr01)
    return matching_files


def process_files():
    # transformed_filtered_messstellen_gw dosyas?n? oku
    transformed_messstellen_nlv_path = 'datasets/transformed_messstellen_nlv.csv'
    df = pd.read_csv(transformed_messstellen_nlv_path)

    # hzbnr01 kodlar?n? al ve string format?na çevirip bo?luklar? temizle
    hzbnr01_codes = df['hzbnr01'].astype(str).str.strip().tolist()

    # processed_rain klasöründeki e?le?en dosyalar? bul
    rain_directory = 'datasets/processed_rain'
    nlv_rain = find_matching_files(hzbnr01_codes, rain_directory)

    # processed_snow klasöründeki e?le?en dosyalar? bul
    snow_directory = 'datasets/processed_snow'
    nlv_snow = find_matching_files(hzbnr01_codes, snow_directory)

    print("E?le?en ya?mur dosyalar?:")
    print(len(nlv_rain))

    print("\nE?le?en kar dosyalar?:")
    print(len(nlv_snow))

    # nlv_rain listesindeki kodlar? içeren sat?rlar? filtrele ve yeni CSV dosyas?na kaydet
    df_rain = df[df['hzbnr01'].astype(str).str.strip().isin(nlv_rain)].reset_index(drop=True)
    df_rain.to_csv('datasets/transformed_messstellen_nlv_rain.csv', index=False)

    # nlv_snow listesindeki kodlar? içeren sat?rlar? filtrele ve yeni CSV dosyas?na kaydet
    df_snow = df[df['hzbnr01'].astype(str).str.strip().isin(nlv_snow)].reset_index(drop=True)
    df_snow.to_csv('datasets/transformed_messstellen_nlv_snow.csv', index=False)


# Fonksiyonu çal??t?r
process_files()
