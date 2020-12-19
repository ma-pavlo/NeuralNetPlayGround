import numpy as np
import h5py
import matplotlib.pyplot as plt
from main import load_cat_data

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


train_x, train_y, test_x, test_y, classes = load_cat_data()

# transform input data / simple transpose
train_x = tf.cast(train_x.T, tf.float32)
train_y =  tf.cast(train_y.T, tf.float32) 

# simple Net in Keras 
model = keras.Sequential()
model.add(layers.Dense(8, activation="relu", input_shape=(12288, ))) # Now model.output_shape is (None, 8), where `None` is the batch dimension. 
model.add(layers.Dense(1, activation="sigmoid"))  

model.summary()

# compile & train
model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_x, train_y, epochs=10, validation_split=0.2, batch_size=209)

