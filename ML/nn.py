import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.regularizers import l1, l2
from keras.layers import Dense



# обучающая выборка с тремя признаками (третий - константа +1)
x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
y_train = [-1, 1, 1, -1, -1, 1, 1, -1, 1, -1]

model = Sequential([
    Dense(10, kernel_regularizer=l1(0.01), activation="relu"),
    Dense(2, activation="softmax")
])

model.compile(optimizer="adam",
              loss=keras.losses.categorical_crossentropy,
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=25, batch_size=32)
