import numpy as np
import h5py
#import random

def load_cat_data():
    """[]

    Returns:
        [type]: [description]
    """
    train_data = h5py.File('data/train_catvnoncat.h5', "r")
    train_x = np.array(train_data["train_set_x"][:])
    train_y = np.array(train_data["train_set_y"][:])

    test_data = h5py.File('data/test_catvnoncat.h5', "r")
    test_x = np.array(test_data["test_set_x"][:])
    test_y = np.array(test_data["test_set_y"][:])

    classes = np.array(test_data["list_classes"][:]) 
    
    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))
    
    return train_x, train_y, test_x, test_y, classes

class BabyNet:
    """[Simple Neural Net; default at 3 layers with each layer at 100 nodes.
        Using Relu activation at each layer except for the final layer, which is a sigmoid by default.
        Default cost function set to squared error cost]
    """
    
    def __init__(
        self,
        layer_size = 100,
        layer_count = 3,
        activation = None, 
        cost=None):

        self.layer_size = layer_size
        self.layer_count = layer_count
        
        # create array holding the activation functions
        if activation == None:
            self.activation = np.append(np.array([relu]*(layer_count-1)), sigmoid)
        else:
            self.activation = activation
        
        # set the cost function
        if cost == None:
            self.cost = squared_error_cost 
        else:
            self.cost = cost

        # load Data 
        self.train_x, self.train_y, self.test_x, self.test_y, self.classes = load_cat_data()

        self.X = self.train_x.reshape(self.train_x.shape[0],-1).T
        self.Y = self.train_y[0]

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
    

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def deriv_sigmoid(z):
    return np.exp(-z)*(1+np.exp(-z)) 

def relu(z):
    return z if z > 0 else 0

def deriv_relu(z):
    return z >= 0  # returns 1 for z > 0; undefined at 0     

def squared_error_cost(a, y):
    return 0.5*(a - y)**2  # wieso 0.5




