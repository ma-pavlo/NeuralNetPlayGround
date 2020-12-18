import numpy as np
import h5py
import matplotlib.pyplot as plt

import keras
import tensorflow as tf

def load_cat_data():
    train_data = h5py.File('data/train_catvnoncat.h5', "r")
    train_x = np.array(train_data["train_set_x"][:])                  # shape: (209, 64, 64, 3)
    train_x = train_x.reshape(train_x.shape[0], -1).T / train_x.max() # shape: (12288, 209)
    train_y = np.array([train_data["train_set_y"][:]])                # shape: (1, 209)

    test_data = h5py.File('data/test_catvnoncat.h5', "r")
    test_x = np.array(test_data["test_set_x"][:])                     # shape: (50, 64, 64, 3)
    test_x = test_x.reshape(test_x.shape[0], -1).T / test_x.max()     # shape: (12288, 50)
    test_y = np.array([test_data["test_set_y"][:]])                   # shape: (1, 50)

    classes = np.array(test_data["list_classes"][:])

    return train_x, train_y, test_x, test_y, classes


train_x, train_y, test_x, test_y, classes = load_cat_data()

train_x = tf.cast(train_x.T, tf.float32)
train_y =  tf.cast(train_y.T, tf.float32) 

# Simple Net in Tensorflow
w1 = tf.Variable(tf.random.normal([12288,8]), np.float32)
b1 = tf.Variable(tf.zeros([1,8]), np.float32)
w2 = tf.Variable(tf.random.normal([8,1]), np.float32)
b2 = tf.Variable(tf.zeros([1,1]), np.float32)


print(tf.reduce_sum([2, 3, 3]))

# Define the model
def model(w1, b1, w2, b2, features=train_x):
	layer1 = tf.keras.activations.relu(tf.matmul(features, w1) + b1)
	return tf.keras.activations.sigmoid(tf.matmul(layer1, w2) + b2)

# Define the loss function
def binary_crossentropy_loss(w1, b1, w2, b2, features=train_x, targets=train_y):
    predictions = model(w1, b1, w2, b2)
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(targets, predictions).numpy()

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
# Train the model
for j in range(2500):
    # Complete the optimizer
	opt.minimize(lambda: binary_crossentropy_loss(w1, b1, w2, b2), 
                 var_list=[w1, b1, w2, b2])

######################################################################
# Simple Net in Keras 
#model = keras.Sequential()
#model.add(keras.layers.Dense(8, activation="relu", input_shape=(12288,)))
#model.add(keras.layers.Dense(1, activation="sigmoid"))
# model.summary()

# Compile & train
#opt = keras.optimizers.Adam(learning_rate=0.01)
#model.compile(loss='binary_crossentropy', optimizer=opt)
#model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# Complete the model fit operation
#model.fit(train_x, train_y, epochs=250, validation_split=0.2)





