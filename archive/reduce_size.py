from PIL import Image
import os


def reduce_image_size(folder_path, max_size_kb=1023):
    # Przejście przez wszystkie pliki w folderze
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path)

            # Proces redukcji rozmiaru pliku
            img_size_kb = os.path.getsize(file_path) / 1024  # rozmiar w KB
            if img_size_kb > max_size_kb:
                factor = (max_size_kb / img_size_kb) ** 0.5  # współczynnik skalowania
                new_width = int(img.width * factor)
                new_height = int(img.height * factor)

                # Skalowanie obrazu
                img = img.resize((new_width, new_height), Image.LANCZOS)

                # Zapisz obraz z próbą dopasowania do docelowego rozmiaru
                img.save(file_path, quality=90, optimize=True)

                # Jeśli obraz nadal jest za duży, próbuj dalej zmniejszać jakość
                # while os.path.getsize(file_path) > max_size_kb * 1024:
                #     img.save(file_path, quality=85, optimize=True)
                #     quality -= 5
                #     if quality < 10:  # Zabezpieczenie, aby nie spadać poniżej akceptowalnej jakości
                #         break

            print(f'Przetworzono obraz: {filename}, Nowy rozmiar: {os.path.getsize(file_path) / 1024:.2f} KB')


# Użycie funkcji
folder_path = 'baza'
reduce_image_size(folder_path)
