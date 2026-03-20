from roboflow import Roboflow
import csv
import os


rf = Roboflow(api_key="Kn62ppAZIXXejSXfHzrE")
project = rf.workspace("butelki").project("defekty_butelek2")
model = project.version(1).model



def create_file(name, arr):
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)

        for i in range(len(arr)):
            writer.writerow(arr[i])
    file.close()

df = []

def extract_details(data):

    details = [0]*12


    for prediction in data['predictions']:
        cls = prediction['class']
        if cls == 'butelka':
            details[0] = prediction['x']
            details[1] = prediction['y']
            details[2] = prediction['width']
            details[3] = prediction['height']
        elif cls == 'etykieta':
            details[4] = prediction['x']
            details[5] = prediction['y']
            details[6] = prediction['width']
            details[7] = prediction['height']
        elif cls == 'korek':
            details[8] = prediction['x']
            details[9] = prediction['y']
            details[10] = prediction['width']
            details[11] = prediction['height']


    if len(details) == 12:  # Upewniamy się, że wszystkie klasy zostały znalezione
      df.append(details)

folder_path = 'temp'

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):  # Sprawdzamy, czy plik ma rozszerzenie .jpg
        file_path = os.path.join(folder_path, filename)  # Pełna ścieżka do pliku
        # Wywołanie funkcji predykcji modelu
        json_data = model.predict(file_path, confidence=40, overlap=30).json()

        # Wywołanie funkcji do przetwarzania danych
        extract_details(json_data)


# json_data = model.predict("your_imag.jpg", confidence=40, overlap=30).json()
# extract_details(json_data)

# visualize your prediction
#model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())







# Definiujesz ścieżkę, gdzie chcesz zapisać plik
file_path = folder_path + ".csv"
create_file(file_path, df)
