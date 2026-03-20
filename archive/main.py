from roboflow import Roboflow
import csv
import numpy as np
import os


# rf = Roboflow(api_key="5qgkrmvLDQhjH9CwX9qa")
# project = rf.workspace().project("defekty_butelek2")
# model = project.version(1).model


rf = Roboflow(api_key="Kn62ppAZIXXejSXfHzrE")
project = rf.workspace("butelki").project("defekty_butelek2")
model = project.version(1).model





#model = get_roboflow_model(model_id="defekty_butelek2/1")

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

    return details


json_data = model.predict("zla_butelka6.jpg", confidence=40, overlap=30).json()
df = extract_details(json_data)
print(df)
result = [
    ["----", "Jest?", "W dobrym miejscu?", "Kształt?"],
    ["Butelka","x","x","x"],
    ["Etykieta","x","x","x"],
    ["Korek","x","x","x"],
]

if df[0] == 0 or df[1] == 0 or df[2] == 0 or df[3] == 0:
    result[1][1] = "x"
else:
    result[1][1] = "v"

if df[4] == 0 or df[5] == 0 or df[6] == 0 or df[7] == 0:
    result[2][1] = "x"
else:
    result[2][1] = "v"

if df[8] == 0 or df[9] == 0 or df[10] == 0 or df[11] == 0:
    result[3][1] = "x"
else:
    result[3][1] = "v"

if (df[0] > 1527*0.98) and (df[0]<1527*1.02) and (df[1] > 1535*0.98) and (df[1]<1535*1.02):
    result[1][2] = "v"
else:
    result[1][2] = "x"

if (df[4] > 1526*0.98) and (df[4]<1526*1.02) and (df[5] > 2127*0.98) and (df[5]<2127*1.02):
    result[2][2] = "v"
else:
    result[2][2] = "x"

if (df[8] > 1487*0.975) and (df[8]<1487*1.025) and (df[9] > 225*0.91) and (df[9]<225*1.09):
    result[3][2] = "v"
else:
    result[3][2] = "x"

if (df[2]>800*0.98) and (df[2]<800*1.02) and (df[3]>2917*0.98) and (df[3]<2917*1.02) and (df[2]*df[3]> 2334687*0.975) and (df[2]*df[3] < 2334687*1.025):
    result[1][3] = "v"
else:
    result[1][3] = "x"

if (df[6]>792*0.98) and (df[6]<792*1.02) and (df[7]>908*0.98) and (df[7]<908*1.02) and (df[6]*df[7]> 718714*0.975) and (df[6]*df[7] < 718714*1.025):
    result[2][3] = "v"
else:
    result[2][3] = "x"

if (df[10]>385*0.975) and (df[10]<385*1.025) and (df[11]>301*0.93) and (df[11]<301*1.15) and (df[10]*df[11]> 115805*0.92) and (df[10]*df[11] < 115805*1.15):
    result[3][3] = "v"
else:
    result[3][3] = "x"

#model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")


print(result)



