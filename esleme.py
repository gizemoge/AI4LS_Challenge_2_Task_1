# ?imdi nlv kodlar? hem ya?mur hem de kar verisini kaps?yor ama ben ikisiniz ayn? de?erlerndirilemeyece?ini dü?ünüyorum o yüzden burda
# onlar? ay?rmay? deniyorum

import os
import pandas as pd




def find_matching_files(hzbnr01_codes, directory):
    matching_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Dosya ad?n?n .csv den önceki son 6 karakterini alarak hzbnr01 kodunu ç?kar
            file_hzbnr01 = filename[-10:-4]  # Son 6 karakteri almak için [-10:-4] aral???n? kullan?yoruz
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

    owf_q_tages = find_matching_files(hzbnr01_codes_owf, 'datasets/Q-Tagesmittel')
    owf_sediment = find_matching_files(hzbnr01_codes_owf, 'datasets/Schwebstoff-Tagesfracht')
    owf_level = find_matching_files(hzbnr01_codes_owf, 'datasets/W-Tagesmittel')
    owf_temp = find_matching_files(hzbnr01_codes_owf, 'datasets/WT-Monatsmittel')

    qu_leit = find_matching_files(hzbnr01_codes_qu, 'datasets/Quellleitfähigkeit-Tagesmittel')
    qu_debi = find_matching_files(hzbnr01_codes_qu, 'datasets/Quellschüttung-Tagesmittel')
    qu_temp = find_matching_files(hzbnr01_codes_qu, 'datasets/Quellwassertemperatur-Tagesmittel')


    print("E?le?en nlv dosyalar?:")
    print(len(nlv_rain))
    print(len(nlv_snow))

    print("E?le?en owf dosyalar?:")
    print(len(owf_q_tages))
    print(len(owf_sediment))
    print(len(owf_level))
    print(len(owf_temp))

    print("E?le?en qu dosyalar?:")
    print(len(qu_leit))
    print(len(qu_debi))
    print(len(qu_temp))



    # nlv_rain listesindeki kodlar? içeren sat?rlar? filtrele ve yeni CSV dosyas?na kaydet
    df_rain = df_nlv[df_nlv['hzbnr01'].astype(str).str.strip().isin(nlv_rain)].reset_index(drop=True)
    df_rain.to_csv('datasets/transformed_nlv_rain.csv', index=False)
    df_snow = df_nlv[df_nlv['hzbnr01'].astype(str).str.strip().isin(nlv_snow)].reset_index(drop=True)
    df_snow.to_csv('datasets/transformed_nlv_snow.csv', index=False)

    df_owf_q_tages = df_owf[df_owf['hzbnr01'].astype(str).str.strip().isin(owf_q_tages)].reset_index(drop=True)
    df_owf_q_tages.to_csv('datasets/transformed_owf_q_debi.csv', index=False)
    df_owf_sediment = df_owf[df_owf['hzbnr01'].astype(str).str.strip().isin(owf_sediment)].reset_index(drop=True)
    df_owf_sediment.to_csv('datasets/transformed_owf_sediment.csv', index=False)
    df_owf_level = df_owf[df_owf['hzbnr01'].astype(str).str.strip().isin(owf_level)].reset_index(drop=True)
    df_owf_level.to_csv('datasets/transformed_owf_level.csv', index=False)
    df_owf_temp = df_owf[df_owf['hzbnr01'].astype(str).str.strip().isin(owf_temp)].reset_index(drop=True)
    df_owf_temp.to_csv('datasets/transformed_owf_temp.csv', index=False)

    df_qu_leit = df_qu[df_qu['hzbnr01'].astype(str).str.strip().isin(qu_leit)].reset_index(drop=True)
    df_qu_leit.to_csv('datasets/transformed_qu_iletkenlik.csv', index=False)
    df_qu_debi = df_qu[df_qu['hzbnr01'].astype(str).str.strip().isin(qu_debi)].reset_index(drop=True)
    df_qu_debi.to_csv('datasets/transformed_qu_debi.csv', index=False)
    df_qu_temp = df_qu[df_qu['hzbnr01'].astype(str).str.strip().isin(qu_temp)].reset_index(drop=True)
    df_qu_temp.to_csv('datasets/transformed_qu_temp.csv', index=False)


# Fonksiyonu çal??t?r
process_files()





