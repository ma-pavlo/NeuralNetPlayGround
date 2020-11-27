import numpy as np
import h5py
import random

class BabyNet:
    
    def __init__(
        self,
        layer_size = 100,
        layer_count = 3,
        activation = None, 
        cost=None):

        self.layer_size = layer_size
        self.layer_count = layer_count
        
        if activation == None:
            self.activation = np.append(np.array([relu]*(layer_count-1)), sigmoid)
        else:
            self.activation = activation
        
        if cost == None:
            self.cost = squared_error_cost 
        else:
            self.cost = cost

        train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

        test_dataset = h5py.File('data/test_catvnoncat.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])

        classes = np.array(test_dataset["list_classes"][:]) 
        
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        
        self.train_set_x_orig = train_set_x_orig
        self.train_set_y_orig = train_set_y_orig
        self.test_set_x_orig = test_set_x_orig
        self.test_set_y_orig = test_set_y_orig
        self.classes = classes

        self.X = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
        self.Y = train_set_y_orig[0]

        self.weights = np.random.rand(layer_count, layer_size, self.X.shape[0])
        self.bias = np.random.rand(layer_count, layer_size, 1)


    def forward_prop(self, x):
        z = []
        a = [x]
        lastZ = x
        lastA = x
        for layer in range(self.layer_count):
            lastZ = self.weights.dot(lastA)+self.bias
            lastA = self.activation[layer](z)
            z.append(lastZ)
            a.append(lastA)
        return (z, a)

    def backward_prop(self):
        # set input layer activations

        #z, a = self.forward_prop(x, w, b)
        return None

        # feed forward: compute z for each layer
        # compute the delta errors in the output
        # in reverse, for each layer calculate the delta errors
        # compute the gradient for updating

        # update the weights and biases
    



#exampleInputLayer = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1)[[0]].T
#exampleOuput = train_set_y_orig[0][0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def deriv_sigmoid(z):
    return np.exp(-z)*(1+np.exp(-z)) #

def relu(z):
    return z if z > 0 else 0

def deriv_relu(z):
    return z if z > 0 else 0 
    #undefined at 0     

# need deriv of sigmoid & relu

def squared_error_cost(a, y):
    return 0.5*(a - y)**2




