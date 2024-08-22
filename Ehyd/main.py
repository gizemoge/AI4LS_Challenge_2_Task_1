# imports
import os
import pandas as pd
import numpy as np


# The selected set of 487 locations in Austria
gw_test_empty = pd.read_csv("Ehyd/datasets/gw_test_empty.csv")

location = list(gw_test_empty.columns[-487:])

def process_datasets(root_directory, type="stand", batch_size=3):
    """
    Her veri seti klasöründeki üçer üçer CSV dosyalarını işleyip ilgili işlenmiş dizinlere kaydeder.

    Args:
        root_directory (str): Veri seti klasörlerini içeren kök dizin.
        type (str): İşlenecek veri seti türü ("stand", "temp", "n", "sh").
        batch_size (int): Her seferinde işlenecek klasör sayısı.
        hzbnr_dict (dict, optional): N-Tagessummen ve SH-Tageswerte dosyaları için kullanılacak eşleme sözlüğü.
    """
    print(f"{root_directory} dizinindeki veri setleri işleniyor...")

    if type == "stand":
        folder_names = sorted([folder_name for folder_name in os.listdir(root_directory) if
                               os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith(
                                   'Grundwasserstand-Monatsmittel-')])
        save_folder = 'processed'

    elif type == "temp":
        folder_names = sorted([folder_name for folder_name in os.listdir(root_directory) if
                               os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith(
                                   "Grundwassertemperatur-Monatsmittel-")])
        save_folder = 'processed_gw_temp'

    elif type == "n":
        folder_names = sorted([folder_name for folder_name in os.listdir(root_directory) if
                               os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith(
                                   "N-Tagessummen")])
        save_folder = 'processed_rain'

    elif type == "sh":
        folder_names = sorted([folder_name for folder_name in os.listdir(root_directory) if
                               os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith(
                                   "SH-Tageswerte")])
        save_folder = 'processed_snow'

        ############################
    elif type == "owf_q_flow_rate":
        folder_names = sorted([folder_name for folder_name in os.listdir(root_directory) if
                               os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith(
                                   "Q-Tagesmittel")])
        save_folder = 'processed_owf_flow_rate'

    elif type == "owf_sediment":
        folder_names = sorted([folder_name for folder_name in os.listdir(root_directory) if
                               os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith(
                                   "Schwebstoff-Tagesfracht")])
        save_folder = 'processed_owf_sediment'

    elif type == "owf_level":
        folder_names = sorted([folder_name for folder_name in os.listdir(root_directory) if
                               os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith(
                                   "W-Tagesmittel")])
        save_folder = 'processed_owf_level'

    elif type == "owf_temp":
        folder_names = sorted([folder_name for folder_name in os.listdir(root_directory) if
                               os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith(
                                   "WT-Monatsmittel")])
        save_folder = 'processed_owf_temp'

    elif type == "qu_flow_rate":
        folder_names = sorted([folder_name for folder_name in os.listdir(root_directory) if
                               os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith(
                                   "Quellschüttung-Tagesmittel")])
        save_folder = 'processed_qu_flow_rate'

    elif type == "qu_conductivity":
        folder_names = sorted([folder_name for folder_name in os.listdir(root_directory) if
                               os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith(
                                   "Quellleitfähigkeit-Tagesmittel")])
        save_folder = 'processed_qu_conductivity'

    elif type == "qu_temp":
        folder_names = sorted([folder_name for folder_name in os.listdir(root_directory) if
                               os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith(
                                   "Quellwassertemperatur-Tagesmittel")])
        save_folder = 'processed_qu_temp'

########################################
    for i in range(0, len(folder_names), batch_size):
        batch_folders = folder_names[i:i + batch_size]

        for folder_name in batch_folders:
            folder_path = os.path.join(root_directory, folder_name)
            processed_folder_path = os.path.join(root_directory, save_folder)

            if not os.path.exists(processed_folder_path):
                os.makedirs(processed_folder_path)

            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)

                    if type in ["stand"]:
                        # 'stand' ve 'temp' türündeki dosyalar için location listesindeki kodları kontrol et
                        if any(str(loc) in file_name for loc in location):
                            output_file_name = f"{file_name.split('.')[0][-6:]}.csv"
                        else:
                            print(f"{file_name} dosyası location listesi ile eşleşmiyor. Bu dosya atlanacak.")
                            continue

                    else:
                        # Diğer türler için standart işleme
                        output_file_name = f"{file_name.split('.')[0][-6:]}.csv"

                    # 'Werte:' kelimesinden sonrasını bul
                    start_index = None
                    with open(file_path, 'r', encoding='windows-1252') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if 'Werte:' in line:
                                start_index = i
                                break

                    if start_index is not None:
                        output_file_path = os.path.join(processed_folder_path, output_file_name)

                        try:
                            # Dosyayı tek seferde oku
                            df = pd.read_csv(file_path, sep=";", skiprows=start_index + 1, encoding='windows-1252')

                            if type in ["stand", "temp", "qu_flow_rate", "qu_conductivity", "qu_temp", "owf_sediment"]:
                                # son sütun ve satırı atarak kaydet (bu dosyalarda boş bir 3.sütun var
                                df.drop(df.columns[-1], axis=1, inplace=True)
                                df.iloc[:-1].to_csv(output_file_path, sep=";", index=False, encoding='windows-1252')
                            else:
                                # 'N-Tagessummen' ve 'SH-Tageswerte' türündeki dosyalar için son sütun silinmez
                                df.iloc[:-1].to_csv(output_file_path, sep=";", index=False, encoding='windows-1252')


                            print(f"{file_name} dosyası {output_file_path} dizinine işlendi.")
                        except pd.errors.ParserError as e:
                            print(f"Hata: {file_name} dosyası işlenirken bir hata oluştu. Bu dosya atlanacak.")
                            continue
                    else:
                        print(f"Hata: {file_name} dosyasında 'Werte:' bulunamadı. Bu dosya atlanacak.")
                else:
                    print(f"{file_name} dosyası listedeki konum numaralarından birini içermiyor. Bu dosya atlanacak.")

# Targetımızı alalım:
root_directory = "Ehyd/datasets"
process_datasets(root_directory, "stand", batch_size=3)



# sıcaklıkları alalım:
process_datasets(root_directory, "temp", batch_size=3)


# şimdi yağmur ve kar verimizi alalım:
process_datasets(root_directory, "n", batch_size=3) #1041 location
process_datasets(root_directory, "sh", batch_size=3) #753 location

####################
process_datasets(root_directory, "owf_q_flow_rate", batch_size=3)
process_datasets(root_directory, "owf_sediment", batch_size=3)
process_datasets(root_directory, "owf_level", batch_size=3)
process_datasets(root_directory, "owf_temp", batch_size=3)
process_datasets(root_directory, "qu_flow_rate", batch_size=3)
process_datasets(root_directory, "qu_conductivity", batch_size=3)
process_datasets(root_directory, "qu_temp", batch_size=3)
######################



# processed lerin in içindeki csvlerde gereksiz boşluklar kaldıralım ve data tipini düzenleyelim
# Lücke yi nan yapalım

def clean_and_save_csv_files(root_directory):
    """
    Verilen kök dizindeki 'processed' ile başlayan her klasördeki CSV dosyalarını işleyip temizlenmiş hallerini üzerine kaydeder.

    Args:
        root_directory (str): Veri seti klasörlerini içeren kök dizin.
    """
    print(f"{root_directory} dizinindeki veri setleri işleniyor...")

    # Kök dizindeki tüm klasörleri al
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)

        # Eğer klasör ismi 'processed' ile başlıyorsa
        if os.path.isdir(folder_path) and folder_name.startswith('processed'):
            # Klasör içindeki tüm CSV dosyalarını işle
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)

                    try:
                        # CSV dosyasını oku
                        if folder_name == 'processed':
                            df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "Target"],
                                             encoding='windows-1252')
                        elif folder_name == 'processed_rain':
                            df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "rain"],
                                             encoding='windows-1252')
                        elif folder_name == 'processed_snow':
                            df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "snow"],
                                             encoding='windows-1252')
                        elif folder_name == 'processed_gw_temp':
                            df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "temp"],
                                             encoding='windows-1252')
                        ##############
                        elif folder_name == 'processed_owf_flow_rate':
                            df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "flow_rate"],
                                             encoding='windows-1252')
                        elif folder_name == 'processed_owf_sediment':
                            df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "sediment"],
                                             encoding='windows-1252')
                        elif folder_name == 'processed_owf_level':
                            df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "level"],
                                             encoding='windows-1252')
                        elif folder_name == 'processed_owf_temp':
                            df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "temp"],
                                             encoding='windows-1252')
                        elif folder_name == 'processed_qu_flow_rate':
                            df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "flow_rate"],
                                             encoding='windows-1252')
                        elif folder_name == 'processed_qu_conductivity':
                            df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "conductivity"],
                                             encoding='windows-1252')
                        elif folder_name == 'processed_qu_temp':
                            df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "temp"],
                                             encoding='windows-1252')
                        #############


                        else:
                            continue  # Diğer klasörler için işlem yapma

                        # Tarih sütununu datetime formatına çevirme
                        df['Date'] = df['Date'].str.strip()

                        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
                        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M', errors='coerce')
                        df['Date'] = df['Date'].fillna(pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce'))
                        df['Date'] = df['Date'].dt.date

                        # Değer sütununun virgülle ayrılan ondalık kısmını nokta ile değiştirme ve float'a dönüştürme
                        value_col = df.columns[1]  # İkinci sütun

                        # Değer sütununun string olduğundan emin ol
                        df[value_col] = df[value_col].astype(str).str.strip()
                        df[value_col] = df[value_col].str.replace(',', '.')
                        # 'Lücke' kelimesini NaN ile değiştirme
                        df[value_col] = df[value_col].replace('Lücke', pd.NA)
                        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

                        # Temizlenmiş dosyayı orijinal dosyanın üzerine kaydet
                        df.to_csv(file_path, sep=';', index=False, encoding='windows-1252')

                        print(f"{file_name} dosyası üzerine kaydedildi.")
                    except Exception as e:
                        print(f"Hata: {file_name} dosyası işlenirken bir hata oluştu: {str(e)}")
                        continue

    print("İşlem tamamlandı.")


clean_and_save_csv_files(root_directory)

# BURADAYIZ




# yağmur ve kar verisi günlük onları aylığa çevirip üstüne kaydedelim:
# bu verilerde nan değerler var ben şimdilik ihmal ettim yani onlar sıfırmış gibi davrandım daha sonra bunu değerlendirip doldurabiliriz

def process_and_resample_files(directory):
    files_with_nan = []

    # Klasördeki tüm dosyaları işle
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)

            try:
                # CSV dosyasını oku
                df = pd.read_csv(file_path, sep=";")

                # 'date' sütununun datetime formatında olduğunu varsay
                df['Date'] = pd.to_datetime(df['Date'])

                # Veriyi aylık toplam olarak yeniden örnekle (NaN değerleri ihmal ederek)
                monthly_df = df.resample('ME', on='Date').sum(min_count=1)  # NaN değerleri ihmal et

                # Aylık veriyi aynı dosyanın üzerine kaydet
                monthly_df.to_csv(file_path, sep=";", index=True)

                print(f"{filename} başarıyla kaydedildi.")
            except Exception as e:
                print(f"{filename} kaydedilemedi. Hata: {e}")


# 'processed_rain' ve 'processed_snow' klasörlerini işleyin
process_and_resample_files('Ehyd/datasets/processed_rain')
process_and_resample_files('Ehyd/datasets/processed_snow')

######
process_and_resample_files('Ehyd/datasets/processed_owf_flow_rate')
process_and_resample_files('Ehyd/datasets/processed_owf_level')
process_and_resample_files('Ehyd/datasets/processed_owf_sediment')

process_and_resample_files('Ehyd/datasets/processed_qu_conductivity')
process_and_resample_files('Ehyd/datasets/processed_qu_flow_rate')
process_and_resample_files('Ehyd/datasets/processed_qu_temp')
########




# messstellen_gw.csv yi ayıklıyoruz:
# CSV dosyasını okuyun
messstellen_gw = pd.read_csv("datasets/messstellen_gw.csv", encoding='windows-1252', delimiter=";")

# hzbnr01 sütunundaki değerlerin location listesindeki elemanlardan biri olup olmadığını kontrol edin
messstellen_gw['hzbnr01'] = messstellen_gw['hzbnr01'].astype(str)
location = [str(item) for item in location]
filtered_messstellen_gw = messstellen_gw[messstellen_gw['hzbnr01'].isin(location)]

# Filtrelenen verileri yeni bir CSV dosyasına kaydedin
filtered_messstellen_gw.to_csv("datasets/filtered_messstellen_gw.csv", index=False)

print(f"Filtrelenen veriler {filtered_messstellen_gw} dosyasına kaydedildi.")
filtered_messstellen_gw = filtered_messstellen_gw.reset_index(drop=True)

# Querleitfähigkeit: malzemenin enine iletkenli?i, suyun içindeki çözünmü? iyonlar?n miktar?n? belirtir.
# Yüksek de?erler, suyun mineral bak?m?ndan zengin oldu?unu veya kirlenmi? olabilece?ini gösterebilir.
