from PIL import Image
import os


def resize_images_in_folder(input_folder, output_folder, size=(1500, 1500)):
    # Sprawdzenie i ewentualne stworzenie folderu wynikowego
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iteracja przez wszystkie pliki w folderze wejściowym
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Sprawdzenie, czy plik jest obrazem (na podstawie rozszerzenia)
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            with Image.open(input_path) as img:
                # Zmiana rozmiaru obrazu
                img = img.resize(size, Image.LANCZOS)  # Używamy metody LANCZOS dla lepszej jakości
                # Zapisywanie zmodyfikowanego obrazu
                img.save(output_path)


# Wywołanie funkcji z odpowiednimi ścieżkami
resize_images_in_folder('temp', 'temp')
