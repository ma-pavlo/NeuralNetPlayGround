import numpy as np
import h5py
#import random

def load_cat_data():
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
        learning_rate = 0.01,
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

        # load data - see dimensions.ipynb
        self.train_x, self.train_y, self.test_x, self.test_y, _ = load_cat_data()

        # reshape data - see dimensions.ipynb
        flat_train_x = self.train_x.reshape(self.train_x.shape[0], -1).T 
        flat_test_x = self.test_x.reshape(self.test_x.shape[0], -1).T
        # standardize data to values between 0 and 1
        self.train_x = flat_train_x/flat_train_x.max()
        self.test_x = flat_test_x/flat_test_x.max()

        # set weights and bias randomly 
        self.layer_dimensions = []
        d = self.initialise_params([2,3,4])
        #self.weights = np.random.randn(layer_count, layer_size, self.train_x.shape[0])
        #self.bias = np.random.randn(layer_count, layer_size, 1)

        
    def initialise_params(layer_dimensions):
        """ Initialise parameter dictionary.

        Args:
            layer_dimensions (python array): contains the dimensions of each layer in the network

        Returns:
            params (python dictionary): contains parameters from W1 & b1 to WL & bL as keys with values to:
                                        W - weight matrix with shape: (layer_dimensions[l], layer_dimensions[l-1])
                                        b - bias vector with shape: (layer_dimensions[l], 1)
        """
        params = {}
        L = len(layer_dimensions)  # amount of layers in the neural net

        for l in range(1, L):
            params['W' + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l-1]) / np.sqrt(layer_dimensions[l-1]) #*0.01
            params['b' + str(l)] = np.zeros((layer_dimensions[l], 1))
        
        return params



#################


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




