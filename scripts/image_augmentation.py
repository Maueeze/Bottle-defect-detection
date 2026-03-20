from PIL import Image, ImageEnhance, ImageFilter
import os
import numpy as np

def enhance_contrast(source_folder, target_folder, contrast_value):

    # Upewnij się, że folder docelowy istnieje. Jeśli nie, utwórz go.
    os.makedirs(target_folder, exist_ok=True)

    # Przejście przez wszystkie pliki w folderze źródłowym.
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.jpg'):
            # Utworzenie nowej nazwy pliku z wartością parametru kontrastu.
            name_part, extension = os.path.splitext(file_name)
            new_file_name = f"{name_part}_contrast_{contrast_value}{extension}"

            # Tworzenie pełnej ścieżki do plików źródłowych i docelowych.
            source_path = os.path.join(source_folder, file_name)
            target_path = os.path.join(target_folder, new_file_name)

            # Otwieranie obrazu, zwiększenie kontrastu i zapisanie do folderu docelowego.
            with Image.open(source_path) as img:
                enhancer = ImageEnhance.Contrast(img)
                enhanced_img = enhancer.enhance(contrast_value)
                enhanced_img.save(target_path, quality=95)

    print(f"Zakończono modyfikację kontrastu zdjęć. Zmodyfikowane zdjęcia zapisano w: {target_folder}")



def convert_to_grayscale(source_folder, target_folder):

    os.makedirs(target_folder, exist_ok=True)

    for file_name in os.listdir(source_folder):
        if file_name.endswith('.jpg'):
            # Utworzenie nowej nazwy pliku z wartością parametru kontrastu.
            name_part, extension = os.path.splitext(file_name)
            new_file_name = f"{name_part}_grey{extension}"

            # Tworzenie pełnej ścieżki do plików źródłowych i docelowych.
            source_path = os.path.join(source_folder, file_name)
            target_path = os.path.join(target_folder, new_file_name)

            with Image.open(source_path) as img:
                grayscale_img = img.convert('L')
                grayscale_img.save(target_path, quality=95)

    print(f"Zakończono konwersję zdjęć na odcienie szarości i zapis w: {target_folder}.")


def gaussian_blur(source_folder, target_folder):

    os.makedirs(target_folder, exist_ok=True)
    radius = 2
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.jpg'):
            # Utworzenie nowej nazwy pliku z wartością parametru kontrastu.
            name_part, extension = os.path.splitext(file_name)
            new_file_name = f"{name_part}_gauss{extension}"

            # Tworzenie pełnej ścieżki do plików źródłowych i docelowych.
            source_path = os.path.join(source_folder, file_name)
            target_path = os.path.join(target_folder, new_file_name)

            with Image.open(source_path) as img:
                blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
                blurred_img.save(target_path, quality=90)

    print(f"Zakończono aplikację filtra Gaussa i zapis")


def salt_and_pepper(source_folder, target_folder, ):

    os.makedirs(target_folder, exist_ok=True)
    noise_level = 0.06
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.jpg'):
            # Utworzenie nowej nazwy pliku z wartością parametru kontrastu.
            name_part, extension = os.path.splitext(file_name)
            new_file_name = f"{name_part}_saltandpepper{extension}"

            # Tworzenie pełnej ścieżki do plików źródłowych i docelowych.
            source_path = os.path.join(source_folder, file_name)
            target_path = os.path.join(target_folder, new_file_name)

            with Image.open(source_path) as img:
                output_image = add_salt_and_pepper_noise(img, noise_level)
                output_image.save(target_path, quality=95)

    print(f"Zakończono aplikację szumu pieprz i sól i zapis.")


def add_salt_and_pepper_noise(image, noise_level):

    # Konwersja obrazu PIL do tablicy NumPy
    np_img = np.array(image)

    # Wprowadzanie szumu
    mask = np.random.choice([0, 1, 2], size=np_img.shape[:2], p=[noise_level / 2, noise_level / 2, 1 - noise_level])
    np_img[mask == 0] = 0  # Sól
    np_img[mask == 1] = 255  # Pieprz

    # Konwersja z powrotem do obrazu PIL
    return Image.fromarray(np_img)




# Przykładowe użycie funkcji:
source = 'D:/butle/chwila'
target = 'D:/butle/zla_butla_gauss'

#enhance_contrast(source, target, 0.75)
#enhance_contrast(source, target, 1.25)
#enhance_contrast(source, target, 1.5)
#convert_to_grayscale(source,target)
gaussian_blur(source, target)
#salt_and_pepper(source, target)

