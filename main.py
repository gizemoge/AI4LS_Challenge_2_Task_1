# imports
import os
import pandas as pd
import numpy as np


# The selected set of 487 locations in Austria
gw_test_empty = pd.read_csv("datasets/gw_test_empty.csv")

location = list(gw_test_empty.columns[-487:])

def process_datasets(root_directory, type="stand", batch_size=3, hzbnr_dict=None):
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
        save_folder = 'processed_temp'

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

                    if type in ["stand", "temp"]:
                        # 'stand' ve 'temp' türündeki dosyalar için location listesindeki kodları kontrol et
                        if any(str(loc) in file_name for loc in location):
                            output_file_name = f"{file_name.split('.')[0][-6:]}.csv"
                        else:
                            print(f"{file_name} dosyası location listesi ile eşleşmiyor. Bu dosya atlanacak.")
                            continue

                    elif type == "n" or type == "sh":
                        # 'N-Tagessummen' ve 'SH-Tageswerte' türündeki dosyalar için hzbnr_dict kontrolü
                        matched_key = None
                        for key, value in hzbnr_dict.items():
                            if str(value) in file_name:
                                matched_key = key
                                break

                        if matched_key:
                            output_file_name = f"{matched_key}.csv"
                        else:
                            print(f"{file_name} dosyası hzbnr_dict değerleri ile eşleşmiyor. Bu dosya atlanacak.")
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

                            if type == "n" or type == "sh":
                                # 'N-Tagessummen' ve 'SH-Tageswerte' türündeki dosyalar için son sütun silinmez
                                df.iloc[:-1].to_csv(output_file_path, sep=";", index=False, encoding='windows-1252')
                            else:
                                # Diğer türlerde son sütun ve satırı atarak kaydet
                                df.drop(df.columns[-1], axis=1, inplace=True)
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
root_directory = "datasets"
process_datasets(root_directory, "stand", batch_size=3)


####################################################################
# sıcaklıkları alalım:
process_datasets(root_directory, "temp", batch_size=3)




# yağmur verisi n-tagessummen(mm) ve kar verisi sh-tagessummen(cm) için bir fonk yazalım, bunlar günlük veriler
# aylığa çevirerek ayıklayalım:

# hzbn eşleşmedi en yakın kooedinatları eşleştirmeyi deneyelimn
# filtered_messstellen_gw ile messstellen.nlv ile koordinatları karşılaştırıp en yakın koordinatları olanların
# hzbnr01 kodlarını eşleştireceğim daha sonra N-Tagessummen ve SH-Tagessummen verilerini bu yeni kodlara göre kullanacağım

filtered_messstellen_gw = pd.read_csv("datasets/filtered_messstellen_gw.csv", encoding='windows-1252')
messstellen_nlv = pd.read_csv("datasets/messstellen_nlv.csv", encoding='windows-1252', delimiter=';')

messstellen_nlv.head()

# ',' yerine '.' ile değiştirme ve float'a çevirme
filtered_messstellen_gw['yhkko10'] = filtered_messstellen_gw['yhkko10'].astype(str).str.replace(',', '.').astype(float)
filtered_messstellen_gw['xrkko09'] = filtered_messstellen_gw['xrkko09'].astype(str).str.replace(',', '.').astype(float)
messstellen_nlv['yhkko09'] = messstellen_nlv['yhkko09'].astype(str).str.replace(',', '.').astype(float)
messstellen_nlv['xrkko08'] = messstellen_nlv['xrkko08'].astype(str).str.replace(',', '.').astype(float)


# Euclidean mesafe hesaplama
def calculate_distance(row, df):
    distances = np.sqrt((df['yhkko09'] - row['yhkko10'])**2 + (df['xrkko08'] - row['xrkko09'])**2)
    return distances
def find_closest_match(row, df):
    distances = calculate_distance(row, df)
    closest_index = distances.idxmin()
    closest = df.loc[closest_index]
    return pd.Series([closest['hzbnr01'], distances.min()], index=['matched_hzbnr01', 'distance'])

filtered_messstellen_gw[['matched_hzbnr01', 'distance']] = filtered_messstellen_gw.apply(lambda row: find_closest_match(row, messstellen_nlv), axis=1)

filtered_messstellen_gw["matched_hzbnr01"] = filtered_messstellen_gw["matched_hzbnr01"].astype(int)



# sözlük oluşturalım:
hzbnr_dict = pd.Series(filtered_messstellen_gw.matched_hzbnr01.values, index=filtered_messstellen_gw.hzbnr01).to_dict()

hzbnr_df = pd.DataFrame(list(hzbnr_dict.items()), columns=['hzbnr', 'nvl_hzbnr'])

# daha sonra kullanmak gerekirse diye kaydedelim:
hzbnr_df.to_csv('datasets/hzbnr.csv', sep=';', index=False)




# şimdi yağmur ve kar verimizi alalım:
process_datasets(root_directory, "n", batch_size=3, hzbnr_dict=hzbnr_dict)
process_datasets(root_directory, "sh", batch_size=3, hzbnr_dict=hzbnr_dict)





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
                        elif folder_name == 'processed_temp':
                            df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "temp"],
                                             encoding='windows-1252')
                        else:
                            continue  # Diğer klasörler için işlem yapma

                        # Tarih sütununu datetime formatına çevirme
                        df['Date'] = df['Date'].str.strip()
                        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

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


# messstellen_gw.csv yi ayıklıyoruz:
# CSV dosyasını okuyun
messstellen_gw = pd.read_csv("datasets/messstellen_gw.csv", encoding='windows-1252', delimiter=";")

# hzbnr01 sütunundaki değerlerin location listesindeki elemanlardan biri olup olmadığını kontrol edin
filtered_messstellen_gw = messstellen_gw[messstellen_gw['hzbnr01'].isin(location)]

# Filtrelenen verileri yeni bir CSV dosyasına kaydedin
filtered_messstellen_gw.to_csv("datasets/filtered_messstellen_gw.csv", index=False)

print(f"Filtrelenen veriler {filtered_messstellen_gw} dosyasına kaydedildi.")
filtered_messstellen_gw = filtered_messstellen_gw.reset_index(drop=True)

# bakıyım bi:

messstellen_nlv = pd.read_csv("datasets/messstellen_nlv.csv", encoding='windows-1252', delimiter=';')

messstellen_nlv.head()

filtered_messstellen_nvl = messstellen_nlv[messstellen_nlv['hzbnr01'].isin(hzbnr_df['nvl_hzbnr'])]

filtered_messstellen_nvl.head()
filtered_messstellen_nvl.to_csv('datasets/filtered_messstellen_nvl.csv', sep=';', index=False)