import pandas as pd #робота с цсв
import numpy as np #математичні операції
import tensorflow as tf #нейронка
from tensorflow import keras #ля тензорфлоу
from tensorflow.keras import layers #для створення шарів
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('Data/figures.csv')
# print(df.head())

encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

X = df[['area', 'perimeter', 'corners']]
y = df['label_enc']

#створення моделі

model = keras.Sequential([layers.Dense(8, activation = "relu", input_shape = (3,)), layers.Dense(8, activation = "relu"), layers.Dense(8, activation = "softmax")])

#навчання

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

history = model.fit(X, y, epochs = 200, verbose = 0)

#візуалізація навчання

plt.plot(history.history['loss'], label = 'Втрата(loss)')
plt.plot(history.history['accuracy'], label = 'Точність(accuracy)')

plt.xlabel("Епоха")
plt.ylabel("Значення")
plt.title("Процес навчання")

plt.legend()
plt.show()

#Тестування

test = np.array([18, 16, 0])

pred = model.predict(test)
print(f'Імовірність по кожному класу: {pred}')
print(f'Модель визначила: {encoder.inverse_transform([np.argmax(pred)])}')