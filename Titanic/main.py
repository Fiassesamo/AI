import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import numpy as np

dataframe = pd.read_csv("titanic.csv")
input_names = ["Age", "Sex", "Pclass"]
output_names = ["Survived"]

raw_input_data = dataframe[input_names]
raw_output_data = dataframe[output_names]

max_age = 100

encoders = {"Age": lambda age: [age/max_age],
            "Sex": lambda gen: {"male": [0], "female": [1]}.get(gen),
            "Pclass": lambda pclass: {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}.get(pclass),
            "Survived": lambda s_value: [s_value]}

def dataframe_to_dict(df):
    result = dict()
    for column in df.columns:
        values = dataframe[column].values
        result[column] = values
    return result

def make_supervised():
    return {"inputs": dataframe_to_dict(raw_input_data),
            "outputs": dataframe_to_dict(raw_output_data)}
def encode(data):
    vectors = []
    for data_name, data_values in data.items():
        encoded = list(map(encoders[data_name], data_values))
        vectors.append(encoded)
    formatted = []
    for vector_raw in list(zip(*vectors)):
        vector = []
        for element in vector_raw:
            for e in element:
                vector.append(e)
        formatted.append(vector)
    return formatted

supervised = make_supervised()
encoded_inputs = np.array(encode(supervised["inputs"]))
encoded_outputs = np.array(encode(supervised["outputs"]))

x_train = encoded_inputs[:600]
x_test = encoded_inputs[600:]

y_train = encoded_outputs[:600]
y_test = encoded_outputs[600:]

model = keras.Sequential([
    Dense(units=64, activation="relu"),
    Dense(units=128, activation="relu"),
    Dense(units=1, activation="sigmoid")
])

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=32)

model.evaluate(x_train, y_train)