# imports
import os
import pandas as pd


# The selected set of 487 locations in Austria
location = [324095, 323295, 323154, 304535, 326934, 307397, 319053, 303727, 319699, 304352, 328419, 328724, 318824,
            329789, 313700, 309617, 301838, 306928, 330928, 325019, 325894, 308585, 321984, 316505, 309211, 327031,
            320754, 329078, 324038, 308668, 304410, 330381, 305060, 310029, 309823, 318444, 322990, 313064, 328690,
            314104, 307843, 323550, 322255, 377887, 316661, 300665, 325928, 301440, 301812, 321471, 331330, 307793,
            302307, 315689, 321778, 304428, 300616, 322610, 328815, 321448, 322115, 304071, 323204, 313833, 304741,
            301937, 327551, 330480, 314534, 302992, 327809, 332783, 312504, 319236, 324020, 328401, 319947, 329813,
            309427, 302380, 323055, 300269, 317230, 327171, 312918, 323253, 316265, 312900, 308924, 323766, 325142,
            325969, 323428, 310268, 319889, 309906, 314161, 319772, 331272, 321430, 313239, 311951, 321646, 302240,
            301846, 300996, 312736, 319764, 305292, 334052, 311639, 323709, 330456, 323097, 330738, 308783, 313569,
            318345, 328773, 329037, 304675, 326595, 312611, 327536, 375113, 330811, 325134, 329649, 302901, 317446,
            303909, 315960, 324327, 328211, 317396, 313643, 319202, 309021, 326975, 314641, 311548, 327619, 323675,
            311944, 307124, 331082, 318873, 313668, 307082, 379313, 313460, 326843, 310862, 331116, 327411, 304733,
            315671, 322578, 323121, 376715, 376608, 313544, 327114, 302588, 319921, 304923, 327239, 328260, 315168,
            331058, 325274, 304170, 323774, 309054, 323618, 305268, 325738, 312165, 329995, 332569, 323410, 376517,
            317461, 331439, 328443, 307520, 300822, 315390, 318485, 320747, 303917, 312660, 330027, 328021, 311266,
            329573, 301309, 330852, 330910, 300236, 375923, 311381, 303263, 314021, 311845, 313817, 304956, 329268,
            322925, 309872, 321752, 315580, 309625, 312447, 300111, 309609, 300400, 308247, 329144, 318584, 310672,
            328435, 300970, 301572, 330274, 307157, 374314, 321950, 376657, 330829, 328666, 376954, 314054, 324434,
            303503, 321836, 321554, 328104, 330803, 322396, 326868, 301127, 304691, 374967, 374074, 309005, 307298,
            304063, 331223, 304600, 329169, 310995, 309419, 301648, 310607, 325167, 309948, 305102, 346056, 322156,
            307769, 316612, 331124, 327437, 319830, 319962, 315853, 329847, 331397, 303016, 328864, 330001, 327163,
            309641, 314294, 323832, 303982, 374678, 322313, 306613, 321992, 333088, 300780, 331298, 310532, 316356,
            317594, 322479, 328302, 307355, 303248, 338616, 300384, 317487, 300137, 305540, 305706, 305714, 305813,
            305821, 305854, 305862, 305896, 305920, 305946, 305953, 306001, 306092, 306183, 306266, 306274, 306399,
            306415, 306456, 306522, 313304, 313338, 313387, 345181, 306043, 305524, 305755, 305904, 305938, 305987,
            305995, 306167, 306209, 305672, 306084, 316026, 316000, 316083, 316091, 316174, 319418, 319426, 319434,
            319541, 319442, 326082, 326108, 326132, 326140, 326074, 326181, 326199, 326223, 326231, 326264, 326249,
            326280, 326298, 326306, 326355, 326371, 326389, 326413, 326439, 326447, 326462, 335067, 335091, 326504,
            335018, 335026, 335208, 335141, 335174, 335182, 335117, 335109, 335216, 335299, 335315, 335323, 335331,
            335349, 335497, 335372, 335414, 335422, 335430, 335448, 335455, 335471, 335521, 335539, 335547, 335554,
            335562, 335570, 335588, 335620, 335638, 335646, 335653, 335661, 335679, 335695, 335778, 335810, 335844,
            335851, 335869, 335596, 335612, 335604, 335877, 335885, 335893, 335927, 335729, 335737, 335828, 335836,
            335901, 335943, 335968, 335976, 335984, 335992, 336008, 345017, 345025, 345041, 345058, 345066, 345108,
            345116, 345124, 345132, 345215, 345173, 345140, 345157, 345165, 345199, 345207, 345249, 345256, 345264,
            345272, 345298, 345314, 345348, 345405, 345397, 345223, 345322, 345330, 345280, 345363, 345371, 345389,
            345421, 345520, 345538, 345546, 345512, 345496, 345504, 345553, 345561, 345579, 345587, 345595, 345413,
            345439, 345447, 345454, 345462, 345470, 345488, 345603, 345660, 345355, 345629, 345710, 345744, 345736,
            345694, 345728, 345645, 345652, 345678, 345686]


def process_datasets(root_directory, type="stand", batch_size=3):
    """
    Her veri seti klasöründeki üçer üçer CSV dosyalarını işleyip ilgili işlenmiş dizinlere kaydeder.

    Args:
        root_directory (str): Veri seti klasörlerini içeren kök dizin.
        batch_size (int): Her seferinde işlenecek klasör sayısı.
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

    for i in range(0, len(folder_names), batch_size):
        batch_folders = folder_names[i:i + batch_size]

        for folder_name in batch_folders:
            folder_path = os.path.join(root_directory, folder_name)
            folder_number = folder_name.split('-')[-1]
            processed_folder_path = os.path.join(root_directory, save_folder)

            if not os.path.exists(processed_folder_path):
                os.makedirs(processed_folder_path)

            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv') and any(str(loc) in file_name for loc in location):
                    file_path = os.path.join(folder_path, file_name)

                    start_index = None
                    with open(file_path, 'r', encoding='windows-1252') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if 'Werte:' in line:
                                start_index = i
                                break

                    if start_index is not None:
                        output_file_name = f"{file_name.split('.')[0][-6:]}.csv"
                        output_file_path = os.path.join(processed_folder_path, output_file_name)

                        try:
                            # Dosyayı tek seferde oku
                            df = pd.read_csv(file_path, sep=";", skiprows=start_index + 1, encoding='windows-1252')

                            # Son sütun ve satırı atarak kaydet
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

# Kullanım
root_directory = "datasets"
process_datasets(root_directory, "stand", batch_size=3)


# processed in içindeki csvlerde gereksiz boşluklar kaldıralım ve data tipini düzenleyelim
# Lücke yi nan yapalım
def clean_and_save_csv_files(root_directory):
    """
    Verilen kök dizindeki her CSV dosyasını işleyip temizlenmiş hallerini üzerine kaydeder.

    Args:
        root_directory (str): Veri seti klasörlerini içeren kök dizin.
    """
    print(f"{root_directory} dizinindeki veri setleri işleniyor...")

    folder_path = os.path.join(root_directory, 'processed')

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            try:
                # CSV dosyasını oku
                df = pd.read_csv(file_path, sep=';', header=None, names=["Date", "Value"], encoding='windows-1252')


                # Tarih sütununu datetime formatına çevirme
                df['Date'] = df['Date'].str.strip()
                df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M:%S')

                # Değer sütununun virgülle ayrılan ondalık kısmını nokta ile değiştirme ve float'a dönüştürme
                df['Value'] = df['Value'].str.strip()
                df['Value'] = df['Value'].str.replace(',', '.')
                # 'Lücke' kelimesini NaN ile değiştirme
                df['Value'] = df['Value'].replace('Lücke', pd.NA)
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

                # Temizlenmiş dosyayı orijinal dosyanın üzerine kaydet
                df.to_csv(file_path, sep=';', index=False, encoding='windows-1252')

                print(f"{file_name} dosyası üzerine kaydedildi.")
            except Exception as e:
                print(f"Hata: {file_name} dosyası işlenirken bir hata oluştu: {str(e)}")
                continue

    print("İşlem tamamlandı.")


# Örnek kullanım:
root_directory = "datasets"
clean_and_save_csv_files(root_directory)


####################################################################
# sıcaklıkları alalım:
root_directory = "datasets"
process_datasets(root_directory, "temp", batch_size=3)






# messstellen_gw.csv yi ayıklıyoruz:
# CSV dosyasını okuyun
messstellen_gw = pd.read_csv("datasets/messstellen_gw.csv", encoding='windows-1252', delimiter=";")

# hzbnr01 sütunundaki değerlerin location listesindeki elemanlardan biri olup olmadığını kontrol edin
filtered_messstellen_gw = messstellen_gw[messstellen_gw['hzbnr01'].isin(location)]

# Filtrelenen verileri yeni bir CSV dosyasına kaydedin
filtered_messstellen_gw.to_csv("datasets/filtered_messstellen_gw.csv", index=False)

print(f"Filtrelenen veriler {filtered_messstellen_gw} dosyasına kaydedildi.")
filtered_messstellen_gw = filtered_messstellen_gw.reset_index(drop=True)
