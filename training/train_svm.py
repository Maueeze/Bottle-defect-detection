import numpy as np
import pandas as pd
from sklearn.svm import SVC
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


def shuffle_array(arr):
    random.shuffle(arr)
    return arr

def summary(data, prediction):

    good_predictions = 0
    for row in range(len(prediction)):
        if prediction[row] == data[row]:
            good_predictions += 1

    efficiency = good_predictions / len(prediction)

    print("\nEffektywność: ", efficiency, "\n")

    confusion_matrix(data, prediction)


def unpack_file(name):
    iris_data = []
    label_data = []
    with open(name) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if len(row) > 0:
                tempiris = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
                templabel = int(row[5])
                iris_data.append(tempiris)
                label_data.append(templabel)

    return iris_data, label_data

def create_file(name, arr):
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)

        for i in range(len(arr)):
            writer.writerow(arr[i])
    file.close()

def confusion_matrix_values(tp, fp, tn, fn, num_class, population):

    feature_value = []
    sensitivity = np.zeros((num_class), dtype=np.float64) #czułość
    specificity = np.zeros((num_class), dtype=np.float64) #swoistość
    precision = np.zeros((num_class), dtype=np.float64) #precyzja
    accuracy = np.zeros((num_class), dtype=np.float64) #dokładność
    f1_score = np.zeros((num_class), dtype=np.float64)

    for i in range(num_class):
            sensitivity[i] = tp[i]/(tp[i]+fn[i])
            specificity[i] = tn[i]/(fp[i]+tn[i])
            precision[i] = tp[i]/(tp[i]+fp[i])
            accuracy[i] = (tp[i]+tn[i])/len(population)
            f1_score[i] = 2 * (precision[i] * sensitivity[i]) / (precision[i] + sensitivity[i])
            k_results[i][0] = sensitivity[i]
            k_results[i][1] = specificity[i]
            k_results[i][2] = precision[i]
            k_results[i][3] = accuracy[i]
            k_results[i][4] = f1_score[i]

    classes = ['0', '1', '2','3','4']
    features = ['Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'F1_Score']

    # wizualizacja macierzy pomyłek
    ax = plt.subplot()
    sns.heatmap(k_results, annot=True, ax=ax, cmap='BuPu', vmin=0.5)

    # ustawienie etykiet osi
    ax.set_title('Dobroć dla algorytmu SVM')
    ax.set_ylabel('Predicted labels')
    ax.set_xlabel('Measures')
    ax.xaxis.set_ticklabels(features)
    ax.yaxis.set_ticklabels(classes)

    # wyświetlenie wykresu
    plt.show()

    for j in range(num_class):
        print("------------------")
        S1 = "Sensitivity for class:", j, "is equal:", sensitivity[j]
        S2 = "Specificity for class:", j, "is equal:", specificity[j]
        S3 = "Precision for class:", j, "is equal:", precision[j]
        S4 = "Accuracy for class:", j, "is equal:", accuracy[j]
        S5 = "F1 score for class:", j, "is equal:", f1_score[j]
        print(S1)
        print(S2)
        print(S3)
        print(S4)
        print(S5)


def confusion_matrix(c_true, c_pred):

    num_classes = 5

    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for i in range(len(c_true)):
        cm[c_true[i], c_pred[i]] += 1

    print("Confusion Matrix:")
    print(cm)

    tp = np.zeros((num_classes), dtype=np.int32)
    fp = np.zeros((num_classes), dtype=np.int32)
    tn = np.zeros((num_classes), dtype=np.int32)
    fn = np.zeros((num_classes), dtype=np.int32)

    for i in range(len(c_true)):
        for j in range(num_classes):
            if ((c_true[i] == j) & (c_pred[i] == j)):
                tp[j] += 1
            if ((c_true[i] != j) & (c_pred[i] == j)):
                fp[j] += 1
            if ((c_true[i] != j) & (c_pred[i] != j)):
                tn[j] += 1
            if ((c_true[i] == j) & (c_pred[i] != j)):
                fn[j] += 1
    print()
    print("TP =", tp)
    print("FP =", fp)
    print("TN =", tn)
    print("FN =", fn)


    confusion_matrix_values(tp, fp, tn, fn, num_classes, c_true)




    classes = ['0', '1', '2','3','4']

    # wizualizacja macierzy pomyłek
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')

    # ustawienie etykiet osi
    ax.set_title('Macierz pomyłek dla algorytmu SVM')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    # wyświetlenie wykresu
    plt.show()


bottle = [ ]
validation_bottle = [ ]
test_bottle = [ ]
name0 = 'data/bez_wad'
name1 = 'data/bez_korka'
name2 = 'data/bez_etykiety'
name3 = 'data/zla_butla'
name4 = 'data/etykieta_misplaced'
with open(name0+'_training_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [ int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 0]
            bottle.append(tempbottle)

with open(name1 + '_training_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 1]
            bottle.append(tempbottle)

with open(name2+'_training_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [ int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 2]
            bottle.append(tempbottle)

with open(name3 + '_training_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 3]
            bottle.append(tempbottle)

with open(name4 + '_training_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 4]
            bottle.append(tempbottle)


with open(name0 + '_validation_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 0]
            validation_bottle.append(tempbottle)

with open(name1+'_validation_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 1]
            validation_bottle.append(tempbottle)

with open(name2 + '_validation_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 2]
            validation_bottle.append(tempbottle)

with open(name3 + '_validation_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 3]
            validation_bottle.append(tempbottle)

with open(name4 + '_validation_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 4]
            validation_bottle.append(tempbottle)




with open(name0 + '_test_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 0]
            bottle.append(tempbottle)

with open(name1+'_test_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 1]
            bottle.append(tempbottle)

with open(name2 + '_test_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 2]
            bottle.append(tempbottle)

with open(name3 + '_test_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 3]
            bottle.append(tempbottle)

with open(name4 + '_test_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 4]
            bottle.append(tempbottle)








shuffle_array(bottle)

X_train = [row[:13] for row in bottle]
y_train = [row[13] for row in bottle]


shuffle_array(validation_bottle)

X_valid = [row[:13] for row in validation_bottle]
y_valid = [row[13] for row in validation_bottle]


shuffle_array(test_bottle)

X_test = [row[:13] for row in test_bottle]
y_test = [row[13] for row in test_bottle]




# X_train, y_train = unpack_file('training_set.csv')
# X_valid, y_valid = unpack_file('validation_set.csv')
# X_test, y_test = unpack_file('test_set.csv')


# Inicjalizacja i dopasowanie modelu SVM
model = SVC(kernel='linear')  # Używamy liniowego jądra SVM
model.fit(X_train, y_train)

k_results = np.zeros((5,5))

pred_labels = model.predict(X_valid)
summary(y_valid,pred_labels)

# pred_labels_test = model.predict(X_test)
# summary(y_test,pred_labels_test)

f1_mean = (k_results[0][4]+k_results[1][4]+k_results[2][4]+k_results[3][4]+k_results[4][4])/5
print(f1_mean)
