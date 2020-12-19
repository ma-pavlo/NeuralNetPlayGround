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
test_x = tf.cast(test_x.T, tf.float32)
test_y =  tf.cast(test_y.T, tf.float32) 

# Simple Net in Tensorflow
inputs = keras.Input(shape=(12288,))
dense = layers.Dense(8, activation="relu")

x = dense(inputs)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="simple_cat_model")

model.summary()

def my_loss_fn(y_true, y_pred):
    binarycrossentr =(- tf.tensordot(y_true, tf.transpose(tf.math.log(y_pred)),1) - tf.tensordot(1-y_true, tf.transpose(tf.math.log(1-y_pred)),1))/ y_true.shape[1]
    return tf.reduce_mean(binarycrossentr, axis=-1)  # the axis=-1


model.compile(optimizer='SGD', loss=my_loss_fn, metrics=["accuracy"])


history = model.fit(train_x, train_y, batch_size=209, epochs=20, validation_split=0.2)

test_scores = model.evaluate(test_x, test_y, verbose=2)
print("Test loss:", test_scores[0])                     # remove [0] when not using accuracy metric
print("Test accuracy:", test_scores[1])                 # remove [0] when not using accuracy metric



keras.utils.plot_model(model, "simple_cat_model.png", show_shapes=True)