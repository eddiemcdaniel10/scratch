# NOTE: use 2.0 beta
# !pip install tensorflow-io-2.0-preview
# import pandas as pd
import numpy as np
import tensorflow as tf

look_back=1
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(4, input_shape=(1, look_back)))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# from sklearn.preprocessing import MinMaxScaler
# import sys

# Replace with PrometheusDataset
dataset = tf.data.experimental.CsvDataset('10min.csv', [tf.int64, tf.float32])


dataset = dataset.map(lambda x, y: y)
datamax = dataset.reduce(0.0, lambda x, y: tf.maximum(x, y))
dataset = dataset.map(lambda x: x/datamax)

print(dataset)

tx = dataset.map(lambda x: tf.reshape(x, [1, 1]))
print(tx)
ty = dataset.skip(1)
print(ty)
tt = tf.data.Dataset.zip((tx, ty)).batch(1)
print(tt)

model.fit(tt, epochs=100)
prediction = model.predict(tt)
for i, pred in enumerate(prediction.flatten()):
    print(f"{i}, {pred * datamax}")