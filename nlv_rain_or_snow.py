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


#####################################


def find_matching_files(hzbnr01_codes, directory):
    matching_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Dosya ad?ndan uzant?y? kald?rarak sadece hzbnr01 kodunu al
            file_hzbnr01 = filename.replace(".csv", "")
            if file_hzbnr01 in hzbnr01_codes:
                matching_files.append(file_hzbnr01)
    return matching_files


def process_and_save(df, hzbnr01_codes, directory, output_filename):
    # Klasördeki e?le?en dosyalar? bul
    matching_files = find_matching_files(hzbnr01_codes, directory)

    # E?le?en kodlar? içeren sat?rlar? filtrele
    df_filtered = df[df['hzbnr01'].astype(str).str.strip().isin(matching_files)].reset_index(drop=True)

    # Yeni CSV dosyas?na kaydet
    df_filtered.to_csv(output_filename, index=False)


def process_files():
    # transformed_filtered_messstellen_gw dosyas?n? oku
    transformed_messstellen_nlv_path = 'datasets/transformed_messstellen_nlv.csv'
    df_nlv = pd.read_csv(transformed_messstellen_nlv_path)
    hzbnr01_codes_nlv = df_nlv['hzbnr01'].astype(str).str.strip().tolist()

    transformed_messstellen_owf_path = 'datasets/transformed_messstellen_owf.csv'
    df_owf = pd.read_csv(transformed_messstellen_owf_path)
    hzbnr01_codes_owf = df_owf['hzbnr01'].astype(str).str.strip().tolist()

    transformed_messstellen_qu_path = 'datasets/transformed_messstellen_qu.csv'
    df_qu = pd.read_csv(transformed_messstellen_qu_path)
    hzbnr01_codes_qu = df_qu['hzbnr01'].astype(str).str.strip().tolist()

    process_and_save(df_nlv, hzbnr01_codes_nlv, 'datasets/N-Tagessummen', 'datasets/transformed_rain.csv')
    process_and_save(df_nlv, hzbnr01_codes_nlv, 'datasets/SH-Tageswerte', 'datasets/transformed_snow.csv')

    process_and_save(df_owf, hzbnr01_codes_owf, 'datasets/Q-Tagesmittel', 'datasets/transformed_owf_q_debi.csv')
    process_and_save(df_owf, hzbnr01_codes_owf, 'datasets/Schwebstoff-Tagesfracht', 'datasets/transformed_owf_st_sediment.csv')
    process_and_save(df_owf, hzbnr01_codes_owf, 'datasets/W-Tagesmittel', 'datasets/transformed_owf_w_suyuksekligi.csv')
    process_and_save(df_owf, hzbnr01_codes_owf, 'datasets/WT-Monatsmittel', 'datasets/transformed_owf_wt_temp.csv')

    process_and_save(df_qu, hzbnr01_codes_qu, 'datasets/Quellleitfähigkeit-Tagesmittel', 'datasets/transformed_qu_leit_iletkenlik.csv')
    process_and_save(df_qu, hzbnr01_codes_qu, 'datasets/Quellschüttung-Tagesmittel', 'datasets/transformed_qu_sch_debi.csv')
    process_and_save(df_qu, hzbnr01_codes_qu, 'datasets/Quellwassertemperatur-Tagesmittel', 'datasets/transformed_qu_temp.csv')

# Fonksiyonu çal??t?r
process_files()

########################################################################################################



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

    transformed_messstellen_nlv_path = 'datasets/transformed_messstellen_nlv.csv'
    df_nlv = pd.read_csv(transformed_messstellen_nlv_path)

    transformed_messstellen_owf_path = 'datasets/transformed_messstellen_owf.csv'
    df_owf = pd.read_csv(transformed_messstellen_owf_path)

    transformed_messstellen_qu_path = 'datasets/transformed_messstellen_qu.csv'
    df_qu = pd.read_csv(transformed_messstellen_qu_path)


    # hzbnr01 kodlar?n? al ve string format?na çevirip bo?luklar? temizle
    hzbnr01_codes_nlv = df_nlv['hzbnr01'].astype(str).str.strip().tolist()
    hzbnr01_codes_owf = df_owf['hzbnr01'].astype(str).str.strip().tolist()
    hzbnr01_codes_qu = df_qu['hzbnr01'].astype(str).str.strip().tolist()

    # e?le?en dosyalar? bul

    nlv_rain = find_matching_files(hzbnr01_codes_nlv, 'datasets/N-Tagessummen')
    nlv_snow = find_matching_files(hzbnr01_codes_nlv, 'datasets/SH-Tageswerte')

    owf_q = find_matching_files(hzbnr01_codes_nlv, 'datasets/Q-Tagesmittel')

#  BURADAYIZ YUKARIDAK? ÇALI?AN FONKU DÖNÜ?TÜRECE??Z ?N?



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


