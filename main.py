import os
import pandas as pd


def process_datasets_with_chunks(root_directory, batch_size=3, chunksize=10000):
    """
    Her veri seti klasöründeki üçer üçer CSV dosyalarını parçalar halinde işleyip ilgili işlenmiş dizinlere kaydeder.

    Args:
        root_directory (str): Veri seti klasörlerini içeren kök dizin.
        batch_size (int): Her seferinde işlenecek klasör sayısı.
        chunksize (int): Her seferinde okunacak satır sayısı.
    """
    print(f"{root_directory} dizinindeki veri setleri işleniyor...")

    folder_names = sorted([folder_name for folder_name in os.listdir(root_directory) if
                           os.path.isdir(os.path.join(root_directory, folder_name)) and folder_name.startswith(
                               'Grundwasserstand-Monatsmittel-')])

    for i in range(0, len(folder_names), batch_size):
        batch_folders = folder_names[i:i + batch_size]

        for folder_name in batch_folders:
            folder_path = os.path.join(root_directory, folder_name)
            folder_number = folder_name.split('-')[-1]
            processed_folder_path = os.path.join(root_directory, f'processed-{folder_number}')

            if not os.path.exists(processed_folder_path):
                os.makedirs(processed_folder_path)

            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)

                    start_index = None
                    with open(file_path, 'r', encoding='latin1') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            if 'Werte:' in line:
                                start_index = i
                                break

                    if start_index is not None:
                        output_file_name = f"{file_name.split('.')[0][-6:]}.csv"
                        output_file_path = os.path.join(processed_folder_path, output_file_name)

                        try:
                            with pd.read_csv(file_path, skiprows=start_index + 1, chunksize=chunksize,
                                             encoding='latin1') as reader:
                                for chunk in reader:
                                    chunk.to_csv(output_file_path, mode='a',
                                                 header=not os.path.exists(output_file_path), index=False)

                            print(f"{file_name} dosyası {output_file_path} dizinine işlendi.")
                        except pd.errors.ParserError as e:
                            print(f"Hata: {file_name} dosyası işlenirken bir hata oluştu. Bu dosya atlanacak.")
                            continue
                    else:
                        print(f"Hata: {file_name} dosyasında 'Werte:' bulunamadı. Bu dosya atlanacak.")


# Kullanım
root_directory = "C:\\Users\\Eda\\PyCharmProjects\\ai4ls\\datasets"
process_datasets_with_chunks(root_directory, batch_size=3)


