import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Wczytanie danych
data_class_0 = pd.read_csv('bez_wad.csv')
data_class_1 = pd.read_csv('bez_korka_gauss.csv')

# Dodanie etykiety klasy do obu zbiorów
data_class_0['label'] = 0
data_class_1['label'] = 1

# Połączenie obu zbiorów w jeden
data = pd.concat([data_class_0, data_class_1], axis=0)

# Przetasowanie danych
data = data.sample(frac=1).reset_index(drop=True)

# Podział danych na cechy i etykiety
X = data.drop('label', axis=1)
y = data['label']

# Podział na zbiory: treningowy, walidacyjny i testowy
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)




# Tworzenie modelu
model = Sequential([
    Dense(64, input_shape=(12,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Kompilacja modelu
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Wyświetlenie podsumowania modelu
model.summary()


# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_val, y_val))


# Ocena modelu na zbiorze testowym
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Accuracy na zbiorze testowym:", test_acc)
