# NOTE: use 2.0 beta
# !pip install tensorflow-io-2.0-preview
# import pandas as pd
import numpy as np
import tensorflow as tf

look_back=10
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(4, input_shape=(1, look_back)))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# from sklearn.preprocessing import MinMaxScaler
# import sys

# Replace with PrometheusDataset

def normalize_data(input_set):
    dataset = input_set.map(lambda x, y: y)
    datamax = dataset.reduce(0.0, lambda x, y: tf.maximum(x, y))
    return dataset.map(lambda x: x/datamax)

dataset = normalize_data(
    tf.data.experimental.CsvDataset('10min.csv', [tf.int64, tf.float32])
    )

test = dataset.batch(10, True)

for x in test:
    print(x)

# tx = dataset.map(lambda x: tf.reshape(x, [1, 1]))
# ty = dataset.skip(1)
# tt = tf.data.Dataset.zip((tx, ty)).batch(1)

# model.fit(tt, epochs=100)
# prediction = model.predict(tt)
# for i, pred in enumerate(prediction.flatten()):
#     print(f"{i}, {pred * datamax}")