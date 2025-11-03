import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense, Flatten


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)
x_test = (x_test.astype("float32") / 255.0)

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([]); plt.yticks([])
    plt.imshow(x_train[i], cmap='gray')

plt.show()

model = keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])


model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=128, epochs=3, validation_split=0.2, verbose=0)
model.evaluate(x_test, y_test_cat)

n = 2
x = np.expand_dims(x_test[n], axis=0) #создание трёхмерного тензера
res = model.predict(x, verbose=0) #сюда нелья передавать матрицы, только трехмерные тензеры
print(f'the number is: {int(np.argmax(res))}')
