import numpy as np
import csv
import random
import seaborn as sns
import matplotlib.pyplot as plt


def shuffle_array(arr):
    random.shuffle(arr)
    return arr


def sign_function(x):
    return 1 if x >= 0 else 0

def perceptron(weights, inputs):
    sum = np.dot(weights, inputs)
    return sign_function(sum)

def perceptron_bez_sign(weights, inputs):
    sum = np.dot(weights, inputs)
    return sum #sign_function(sum)

def confusion_matrix_values(tp, fp, tn, fn, num_class, population):
    f1_mean = 0
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
            f1_score[i] = 2*(precision[i]*sensitivity[i])/(precision[i]+sensitivity[i])
            f1_mean = f1_mean + f1_score[i]

    for j in range(num_class):
        print("------------------")
        print("Sensitivity for class:", j, "is equal:", sensitivity[j])
        print("Specificity for class:", j, "is equal:", specificity[j])
        print("Precision for class:", j, "is equal:", precision[j])
        print("Accuracy for class:", j, "is equal:", accuracy[j])
        print("F1 score for class:", j, "is equal:", f1_score[j])

    print(f1_mean/5)


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
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    # wyświetlenie wykresu
    plt.show()


bottle= [ ]
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



shuffle_array(bottle)

bottle_data = [row[:13] for row in bottle]
label_data = [row[13] for row in bottle]
print(bottle_data[1:10])
print(label_data[1:10])




#print(iris_data[1:10])
print('klasy danych testowych:', label_data[0:10])

# Trenowanie modelu one vs all
learning_rate = 0.15
epochs = 1000
n_classes = 5
weights = np.zeros((n_classes, 13))

print('-------')
print('podział:')
for i in range(n_classes):
    y_i = np.zeros(len(label_data))
    for j in range(len(label_data)):
        if label_data[j] == i:
            ty_i = 1
        else:
            ty_i = 0
        y_i[j] = ty_i

    print(y_i[0:10])
    for _ in range(epochs):
        for xi, target in zip(bottle_data, y_i):
            delta_w = learning_rate * (target - perceptron(weights[i], xi))
            for k in range(len(xi)):
                weights[i][k] += delta_w * xi[k]


# Wyświetlenie wag
print('Wagi: ', weights)

validation_bottle = [ ]

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
            validation_bottle.append(tempbottle)

with open(name1+'_test_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 1]
            validation_bottle.append(tempbottle)

with open(name2 + '_test_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 2]
            validation_bottle.append(tempbottle)

with open(name3 + '_test_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 3]
            validation_bottle.append(tempbottle)

with open(name4 + '_test_set.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if len(row) > 0:
            tempbottle = [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), 4]
            validation_bottle.append(tempbottle)


shuffle_array(validation_bottle)

validation_bottle_data = [row[:13] for row in validation_bottle]
validation_label_data = [row[13] for row in validation_bottle]



pred_labels = []
#Przykładowe przewidywanie klasy dla nowych danych
for i in range(len(validation_bottle_data)):
    y_pred = []
    #print("numer:", i)
    for j in range(n_classes):

        suma = perceptron(weights[j], validation_bottle_data[i])
        #print(suma)
        if suma == 1:

            y_pred.append(j)

    y_pred = shuffle_array(y_pred)
    if not y_pred:
        y = random.randrange(5)
        y_pred.append(y)
    pred_labels.append(y_pred[0])



print(pred_labels)
print(validation_label_data)


good_predictions = 0
for row in range(len(pred_labels)):
    if pred_labels[row] == validation_label_data[row]:
        good_predictions += 1

efficiency = good_predictions/len(pred_labels)

print("\nEffektywność: ", efficiency, "\n")

confusion_matrix(validation_label_data, pred_labels)

#
#np.savetxt('wagi2.csv', weights, delimiter=',', fmt='%0.7f')