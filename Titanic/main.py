import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

dataframe = pd.read_csv("titanic.csv")
input_names = ["Age", "Sex", "Pclass"]
output_names = ["Survived"]

raw_input_data = dataframe[input_names]
output_input_data = dataframe[output_names]

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

def make_supervised(df):
    raw_input_data = dataframe[input_names]
    raw_output_data = dataframe[output_names]
    return {"inputs": dataframe_to_dict(raw_input_data),
            "outputs": dataframe_to_dict(raw_output_data)}


def encode(data):
    vectors = []
    for data_names, data_values in data.items():
        encoded = list(map(encoders[data_names], data_values))
        vectors.append(encoded)
    formatted = []
    for vector_raw in list(zip(*vectors)):
        vector = []
        for element in vector_raw:
            for e in element:
                vector.append(e)
        formatted.append(vector)
    return formatted


supervised = make_supervised(dataframe)
encoded_inputs = encode(supervised["inputs"])
encoded_outputs = encode(supervised["outputs"])


x_train = encoded_inputs[:600]
y_train = encoded_outputs[:600]

x_test = encoded_inputs[600:]
y_test = encoded_outputs[600:]

model = keras.Sequential([
    Dense(units=5, activation="relu"),
    Dense(units=1, activation="softmax")
])

model.compile(optimizer="adam",
              )