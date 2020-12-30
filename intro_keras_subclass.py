# https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/

from keras.layers import Dense, Input
import keras
from main import load_cat_data
import tensorflow as tf


class Subclass(keras.Model):

    def __init__(self):
        super(Subclass, self).__init__(name='simple_net')

        self.dense_1 = Dense(8, activation='relu')
        self.dense_2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)
    
    # Could not print model summary - https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model
    # took the ugly solution:
    def summary(self):
        x = Input(shape=(12288))
        model = keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


model = Subclass()

model.summary()

# compile
model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss='binary_crossentropy')#, metrics=['accuracy'])

# load data
train_x, train_y, test_x, test_y, classes = load_cat_data()

# transform input data / simple transpose
train_x = tf.cast(train_x.T, tf.float32)
train_y =  tf.cast(train_y.T, tf.float32) 

# train the model
model.fit(train_x, train_y, epochs=10, validation_split=0.2, batch_size=209)
