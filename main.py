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
    """Simple Neural Net; default at 3 layers with each layer at 100 nodes.
        Using Relu activation at each layer except for the final layer, which is a sigmoid by default.
        Default cost function set to squared error cost.
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
        self.learning_rate = learning_rate,
        
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

        pixel_num = self.train_x.shape[1]
        # initialise parameters (weights and bias) randomly within a range of -1 to 1 for the weights, bias can be zero initialisation
        
        layer_dimensions = np.array([3,100])
        self.params = self.initialise_params(layer_dimensions)
        #self.weights = np.random.randn(layer_count, layer_size, self.train_x.shape[0])
        #self.bias = np.random.randn(layer_count, layer_size, 1)

#######################################################
        #self.cost_track = [] # save costs to check updating        
        # 1. Initialise params --- initialise_params(self, layer_dimensions)
        
        # 2. Loop thorugh Full BAtch Grad descent
        #for i in range(0, 100):
            # 3. forwardProp
            # AL, caches = forward_prop(self, X, parameters)
        
            # 4. calculate cost
            # cost = costFunction()
    
            # 5. BackProp propagation.
            # gradients = backward_prop(AL, Y, caches)
 
            # 6. update params
            #params = update_params()
                
        #print cost (should be changed to not show at each iteration but less)
        #print(cost)
        #self.cost_track.append(cost)
#######################################################

    

        
    def initialise_params(self, layer_dimensions):
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
            params['W' + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l-1])*0.01
            params['b' + str(l)] = np.zeros((layer_dimensions[l], 1))
        return params
    
    def update_params(self, params, gradients, learning_rate):
        L = len(params) // 2 
        for l in range(L):
            params["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate * gradients["dW" + str(l+1)]
            params["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate * gradients["db" + str(l+1)]
        return params

    def forward_steps(self, A_prev, W, b, activation):
        """ Compute Linear step then Activation Function.

        Args:
            A_prev ([type]): [description]
            W ([type]): [description]
            b ([type]): [description]
            activation ([type]): [description]

        Returns:
            [type]: [description]
        """
        if activation == "sigmoid":
            Z = W.dot(A_prev) + b
            linear_cache = (A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        
        elif activation == "relu":
            Z = W.dot(A_prev) + b
            linear_cache = (A_prev, W, b)
            A, activation_cache = relu(Z)
        
        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_prop(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2   # num of layers in the network //2 due to Wb
        
        # linear -> relu * (the number of defined layers - 1)
        for l in range(1, L):
            A, cache = self.forward_steps(A, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
            caches.append(cache)
        
        # linear -> sigmoid (final layer)
        AL, cache = self.forward_steps(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
        caches.append(cache)
        
        return AL, caches

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




