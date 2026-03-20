import csv
import random

def shuffle_array(arr):
    random.shuffle(arr)
    return arr

def split_array(arr):
    n = len(arr)
    split_index1 = int(0.7 * n)
    split_index2 = int(0.85 * n)
    arr1 = arr[:split_index1]
    arr2 = arr[split_index1:split_index2]
    arr3 = arr[split_index2:]
    return arr1, arr2, arr3

def create_file(name, arr):
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)

        for i in range(len(arr)):
            writer.writerow(arr[i])
    file.close()

bottle_data = []
file_name ='bez_wad'
a=3000
with open(file_name+".csv") as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        if len(row) > 0:
            tempbottle = [1, float(row[0])/a, float(row[1])/a, float(row[2])/a, float(row[3])/a, float(row[4])/a, float(row[5])/a, float(row[6])/a, float(row[7])/a, float(row[8])/a, float(row[9])/a, float(row[10])/a, float(row[11])/a, 0]
            bottle_data.append(tempbottle)


shuffle_array(bottle_data)
training_set_with_labels, validation_set_with_labels, test_set_with_labels = split_array(bottle_data)

print(len(training_set_with_labels))
print(len(validation_set_with_labels))
print(len(test_set_with_labels))

create_file(file_name+"_training_set.csv", training_set_with_labels)
create_file(file_name+"_validation_set.csv", validation_set_with_labels)
create_file(file_name+"_test_set.csv", test_set_with_labels)