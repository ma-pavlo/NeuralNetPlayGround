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

# # load data - see dimensions.ipynb
# self.train_x_raw, self.train_y, self.test_x_raw, self.test_y, _ = load_cat_data()

# # reshape data - see dimensions.ipynb
# flat_train_x = self.train_x_raw.reshape(self.train_x_raw.shape[0], -1).T
# flat_test_x = self.test_x_raw.reshape(self.test_x_raw.shape[0], -1).T
# # standardize data to values between 0 and 1
# self.train_x = flat_train_x/flat_train_x.max()
# self.test_x = flat_test_x/flat_test_x.max()

class BabyNet:
    """Simple Neural Net; default at 2 layers with each hiddem layer at 8 nodes and the final layer at 1 node.
        Using Relu activation at each layer except for the final layer, which is a sigmoid by default.
        Default cost function set to squared error cost.
    """

    def __init__(
        self,
        layer_dimensions = [12288,8,1],
        activation = None,
        activation_deriv = None,
        cost = None,
        cost_deriv = None):

        self.layer_dimensions = layer_dimensions

        # create array holding the activation functions
        if activation == None:
            self.activation = np.append(np.array([relu]*(len(layer_dimensions)-1)), sigmoid)
            self.activation_deriv = np.append(np.array([relu_deriv]*(len(layer_dimensions)-1)), sigmoid_deriv)
        else:
            self.activation = activation
            self.activation_deriv = activation_deriv

        # set the cost function
        if cost == None:
            self.cost = squared_error_cost
            self.cost_deriv = squared_error_cost_deriv
        else:
            self.cost = cost
            self.cost_deriv = cost_deriv

        self.params = self.initialise_params(layer_dimensions)


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
                                            and values between -0.01 and 0.01
                                        b - bias vector with shape: (layer_dimensions[l], 1)
                                            and values equal to 0
        """
        params = {}
        L = len(layer_dimensions)  # amount of layers in the neural net
        for l in range(1, L):
            params['W' + str(l-1)] = np.random.randn(layer_dimensions[l], layer_dimensions[l-1])*0.01
            params['b' + str(l-1)] = np.zeros((layer_dimensions[l], 1))
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
        """
        Args:
            x (vector of length layer_dimensions[0]): contains the values for the input layer

        Returns:
            z is an array of size len(layer_dimensions) where each i'th element is a vector of size layer_dimensions[i]
            a is just z with the appropriate activation function for each layer applied
        """
        z = []
        a = [x]
        lastA = x
        for layer in range(0, len(self.layer_dimensions) - 1):
            weights = self.params['W' + str(layer)]
            bias = self.params['b' + str(layer)]
            lastZ = (weights @ lastA) + bias
            lastA = self.activation[layer](lastZ)
            z.append(lastZ)
            a.append(lastA)
        return z, a

    def train_one_piece_of_data(self, x, y, learning_rate = 0.01):
        # feed forward
        z, a = self.forward_prop(x)

        # using the maths from http://neuralnetworksanddeeplearning.com/chap2.html
        L = len(self.layer_dimensions) - 2
        # compute the delta errors in the output
        d = [np.multiply(self.cost_deriv(a[L+1], y), self.activation_deriv[L](z[L]))]

        # back propagation: in reverse, for each layer calculate the delta errors
        # looks weird because this is a backwards loop, but it goes from [0 to L - 2] inclusive
        for l in range(L - 1, -1, -1):
            w = self.params['W' + str(l+1)]
            arg1 = w.T @ d[0]
            arg2 = self.activation_deriv[l](z[l])
            d = [np.multiply(arg1, arg2)] + d

        # update the weights and biases
        for l in range(0, len(self.layer_dimensions) - 1):
            w = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            print('update', l, w.shape, a[l].shape, d[l].shape)
            deltaW = a[l] * d[l]
            deltaB = d[l]
            self.params['W' + str(l)] = w - learning_rate * deltaW
            self.params['b' + str(l)] = b - learning_rate * deltaB


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    return np.exp(-z)*(1+np.exp(-z))

def relu(z):
    return z * (z > 0)

def relu_deriv(z):
    return z >= 0  # returns 1 for z > 0; undefined at 0

def squared_error_cost(a, y):
    """
        Args:
            a is a vector of output predictions
            y is a vector of training data outputs

        Returns:
            a vector where each element is the squared difference
    """
    return 0.5*(a - y)**2

def squared_error_cost_deriv(a, y):
    return (a - y)




train_x, train_y, test_x, test_y, classes = load_cat_data()
# reshape the training and test examples
train_x_flat = train_x.reshape(train_x.shape[0], -1).T
test_x_flat = test_x.reshape(test_x.shape[0], -1).T

# standardize data to values between 0 and 1
standardized_train_x = train_x_flat/train_x_flat.max()
standardized_test_x = test_x_flat/train_x_flat.max()
net = BabyNet([12288,8,1])

for x, y in zip(standardized_train_x.T, train_y.T):
    net.train_one_piece_of_data(np.array([x]).T, np.array([y]).T)

    # accuracy:
    totalTrainError = 0
    totalTestError = 0
    for trainX, trainY in zip(standardized_train_x.T, train_y.T):
        # print(net.forward_prop(trainX).shape)
        z, a = net.forward_prop(np.array([trainX]))
        #print(len(a))
        #print(a[0].shape)
        #print(a[1].shape)
        #print(a[2].shape)
        error = trainY[0] - a
        totalTrainError += error
    #for testX, testY in zip(standardized_test_x.T, test_y.T):
    #    error = testY[0] - net.forward_prop(testX)
    #    totalTestError += error

    #testAccuracy = totalTestError / len(test_y)
    trainAccuracy = totalTrainError / len(train_y)
    #print('Train Acc: ', trainAccuracy)