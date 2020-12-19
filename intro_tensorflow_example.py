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



# # Simple Net in Tensorflow
# w1 = tf.Variable(tf.random.normal([12288,8]), np.float32)
# b1 = tf.Variable(tf.zeros([1,8]), np.float32)
# w2 = tf.Variable(tf.random.normal([8,1]), np.float32)
# b2 = tf.Variable(tf.zeros([1,1]), np.float32)


# print(tf.reduce_sum([2, 3, 3]))

# # Define the model
# def model(w1, b1, w2, b2, features=train_x):
# 	layer1 = tf.keras.activations.relu(tf.matmul(features, w1) + b1)
# 	return tf.keras.activations.sigmoid(tf.matmul(layer1, w2) + b2)

# # Define the loss function
# def binary_crossentropy_loss(w1, b1, w2, b2, features=train_x, targets=train_y):
#     predictions = model(w1, b1, w2, b2)
#     bce = tf.keras.losses.BinaryCrossentropy()
#     return bce(targets, predictions).numpy()

# opt = tf.keras.optimizers.Adam(learning_rate=0.01)
# # Train the model
# for j in range(2500):
#     # Complete the optimizer
# 	opt.minimize(lambda: binary_crossentropy_loss(w1, b1, w2, b2), 
#                  var_list=[w1, b1, w2, b2])





# KERAS :


# (x_train, y_train), (x_test, y_test) = mnist.load_data()


# x_train = x_train.reshape(-1, 784)
# x_test = x_test.reshape(-1, 784)
# x_train = x_train.astype("float32")
# x_test = x_test.astype("float32")
# x_train /= 255
# x_test /= 255
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# # model = keras.Sequential()
# # model.add(layers.Dense(8, activation='relu', input_shape=(784, )))
# # model.add(layers.Dense(1, activation='sigmoid'))

# # model.summary()