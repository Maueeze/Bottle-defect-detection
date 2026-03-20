from roboflow import Roboflow
import csv
import numpy as np
import random
import os
from PIL import Image, ImageDraw, ImageFont

rf = Roboflow(api_key="----------")
project = rf.workspace("butelki").project("defekty_butelek2")
model = project.version(1).model

#model = get_roboflow_model(model_id="defekty_butelek2/1")
image_size = 1500
a = image_size

def resize_image(input_path, output_path, size=(a, a)):
    # Otwieranie obrazu
    with Image.open(input_path) as img:
        # Zmiana rozmiaru obrazu
        img = img.resize(size, Image.LANCZOS)  # Używamy metody LANCZOS dla lepszej jakości
        # Zapisywanie zmodyfikowanego obrazu
        img.save(output_path)


def create_file(name, arr):
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)

        for i in range(len(arr)):
            writer.writerow(arr[i])
    file.close()

df = []
def extract_details(data):

    details = [0]*13


    for prediction in data['predictions']:
        cls = prediction['class']
        if cls == 'butelka':
            details[0] = 1
            details[1] = prediction['x']/a
            details[2] = prediction['y']/a
            details[3] = prediction['width']/a
            details[4] = prediction['height']/a
        elif cls == 'etykieta':
            details[5] = prediction['x']/a
            details[6] = prediction['y']/a
            details[7] = prediction['width']/a
            details[8] = prediction['height']/a
        elif cls == 'korek':
            details[9] = prediction['x']/a
            details[10] = prediction['y']/a
            details[11] = prediction['width']/a
            details[12] = prediction['height']/a

    return details

def sign_function(x):
    return 1 if x >= 0 else 0

def perceptron(weights, inputs):
    sum = np.dot(weights, inputs)
    return sign_function(sum)

def draw_bounding_box(image_path, output_path, box_coordinates, label, color):

    # Wczytanie obrazu
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)

        # Wyciągnięcie współrzędnych
        x, y, width, height = box_coordinates
        top_left = (x-width/2, y - height/2)
        bottom_right = (x + width/2, y + height/2)

        # Rysowanie prostokąta
        draw.rectangle([top_left, bottom_right], outline=color, width=4)
        size = 35
        # Dodanie etykiety
        try:
            # Spróbuj załadować font (może nie działać na niektórych środowiskach bez dodatkowych fontów)
            font = ImageFont.truetype("arial.ttf", size)
        except IOError:
            # Użyj domyślnego fontu, jeśli niemożliwe jest załadowanie "arial.ttf"
            font = ImageFont.load_default()

        text_position = (x - width/2 + 15, y - height/2 + 15)
        draw.text(text_position, label, fill=color, font=font)

        # Zapisanie obrazu
        img.save(output_path)





file_name = "bez_korka.jpg"
nazwa_pliku = 'resized_file.jpg'

resize_image(file_name, nazwa_pliku)

json_data = model.predict(nazwa_pliku, confidence=40, overlap=30).json()
df = extract_details(json_data)
print(df)





csv_file_path = 'wagi.csv'
# Wczytywanie tablicy numpy z pliku CSV
weights = np.loadtxt(csv_file_path, delimiter=',')


pred_labels = np.zeros(5)
for j in range(5):

    suma = perceptron(weights[j], df)
    # print(suma)
    if suma > 0:
        pred_labels[j] = suma

print(pred_labels)


nazwa_pliku_wyjsciowego = 'prediction_pillow.jpg'

if df[1]*df[2]*df[3]*df[4]!= 0:
    draw_bounding_box(nazwa_pliku, nazwa_pliku_wyjsciowego, (df[1]*a, df[2]*a, df[3]*a, df[4]*a), 'Butelka','red')
if df[5]*df[6]*df[7]*df[8]!= 0:
    draw_bounding_box(nazwa_pliku_wyjsciowego, nazwa_pliku_wyjsciowego, (df[5]*a, df[6]*a, df[7]*a, df[8]*a), 'Etykieta','green')
if df[9]*df[10]*df[11]*df[12]!= 0:
    draw_bounding_box(nazwa_pliku_wyjsciowego, nazwa_pliku_wyjsciowego, (df[9]*a, df[10]*a, df[11]*a, df[12]*a), 'Korek','blue')


#model.predict(nazwa_pliku, confidence=40, overlap=30).save("prediction.jpg")






