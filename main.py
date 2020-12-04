import numpy as np
import h5py

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

class BabyNet:
    """
        Simple Neural Net; default at 2 layers with each hiddem layer at 8 nodes and the final layer at 1 node.
        Using Relu activation at each layer except for the final layer, which is a sigmoid by default.
        Default cost function set to squared error cost.
    """

    def __init__(
        self,
        layer_dimensions = [12288,8,1],
        activations = None,
        activation_derivs = None,
        cost = None,
        cost_deriv = None):
        """
        Args:
            layer_dimensions: list<integer>
                - the number of nodes for each layer, including the input and output layers. Has length L.
            activations: list<(z: array<layer_dimensions[i],1>) => array<layer_dimensions[i],1>>
                - the activation function to use for each layer. Has length L, the entry at index 0 is not used.
            activation_derivs: list<(z: array<layer_dimensions[i],1>) => array<layer_dimensions[i],1>>
                - the activation function derivative to use for each layer. Has length L, the entry at index 0 is not used.
            cost: (a: array<layer_dimensions[L-1],1>, y: array<layer_dimensions[L-1],1>) => array<layer_dimensions[L-1],1>
                - the cost function. Takes a vector 'a' with the activations from the output layer, and a vector 'y' with the training data outputs, and produces a vector with the errors.
            cost_deriv: (a: array<layer_dimensions[L-1],1>, y: array<layer_dimensions[L-1],1>) => array<layer_dimensions[L-1],1>
                - the cost function derivative.
        """

        if activations == None:
            self.activations = np.append(np.array([relu]*(len(layer_dimensions)-1)), sigmoid)
            self.activation_derivs = np.append(np.array([relu_deriv]*(len(layer_dimensions)-1)), sigmoid_deriv)
        else:
            self.activations = activations
            self.activation_derivs = activation_derivs

        # set the cost function
        if cost == None:
            self.cost = binary_cross_entropy
            self.cost_deriv = binary_cross_entropy_deriv
        else:
            self.cost = cost
            self.cost_deriv = cost_deriv

        self.layer_dimensions = layer_dimensions
        self.L = len(layer_dimensions)

        # initialize with None at index 0, as there are no weights or biases for layer 0
        self.weights = [None]
        self.biases = [None]
        for l in range(1, self.L):
            # a matrix with layer_dimensions[l] rows and layer_dimensions[l-1] columns
            self.weights.append(np.random.randn(layer_dimensions[l], layer_dimensions[l-1])*0.01)
            # a vector with layer_dimensions[l] rows
            self.biases.append(np.zeros((layer_dimensions[l], 1)))

    def forward_prop(self, x):
        """
        Args:
            x: array<layer_dimensions[0],N>
                - training data inputs. Where N is the batch size.

        Returns:
            z: list<array<layer_dimensions[i],N>>
                - the calculated z for each layer. Has length L.
            a: list<array<layer_dimensions[i],N>>
                - the calculated activations for each layer (this is just z with the appropriate activation function for each layer applied). Has length L.
        """
        z = [x]
        a = [x]
        for l in range(1, self.L):
            currentZ = self.weights[l] @ a[l-1] + self.biases[l]
            z.append(currentZ)
            a.append(self.activations[l](currentZ))
        return z, a

    def train_one_piece_of_data(self, x, y, learning_rate = 0.0075):
        """
        Args:
            x: array<layer_dimensions[0],1>
            y: array<layer_dimensions[L-1],1>
            learning_rate: number
        """
        # using the maths from http://neuralnetworksanddeeplearning.com/chap2.html

        L = self.L

        # feed forward
        z, a = self.forward_prop(x)

        # compute the delta errors in the output
        d = {
            (L-1): self.cost_deriv(a[L-1], y) * self.activation_derivs[L-1](z[L-1])
        }

        # back propagation: in reverse, for each layer calculate the delta errors
        # this is a backwards loop, it goes from L - 2 to 0 inclusive
        for l in range(L - 2, 0, -1):
            d[l] = (self.weights[l+1].T @ d[l+1]) * self.activation_derivs[l](z[l])

        # update the weights and biases
        for l in range(1, L):
            # row j, col k of dw should be = a[l-1][k] * d[l][j]
            dw = a[l-1].T * d[l]
            db = d[l]
            self.weights[l] = self.weights[l] - learning_rate * dw
            self.biases[l] = self.biases[l] - learning_rate * db

    def train_batch(self, x, y, learning_rate = 0.001):
        """
        Args:
            x: array<layer_dimensions[0],N>
                - where N is the batch size.
            y: array<layer_dimensions[L-1],N>
            learning_rate: number
        """
        # using the maths from http://neuralnetworksanddeeplearning.com/chap2.html

        L = self.L
        N = y.shape[1]
        z, a = self.forward_prop(x)

        # compute the delta errors in the output
        d = {
            (L-1): self.cost_deriv(a[L-1], y) * self.activation_derivs[L-1](z[L-1])
        }

        # back propagation: in reverse, for each layer calculate the delta errors
        # this is a backwards loop, it goes from L - 2 to 0 inclusive
        for l in range(L - 2, 0, -1):
            d[l] = (self.weights[l+1].T @ d[l+1]) * self.activation_derivs[l](z[l])

        # update the weights and biases
        for l in range(1, L):
            for n in range(0, N):
                # row j, col k of dw should be = a[l-1][k] * d[l][j]
                dw = a[l-1][:,[n]].T * d[l][:,[n]]
                db = d[l][:, [n]]
                self.weights[l] = self.weights[l] - learning_rate * dw
                self.biases[l] = self.biases[l] - learning_rate * db



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
            a: array<n,1>
                - activations from the final layer.
            y: array<n,1>
                - training data outputs.

        Returns:
            : number
                - the squared differences.
    """
    return np.sum(0.5*(a - y)**2)/len(y)

def squared_error_cost_deriv(a, y):
    return (a - y)

def binary_cross_entropy(a, y):
    """
        Args:
            a: array<layer_dimensions[L-1], N>
                - activations from the final layer.
            y: array<layer_dimensions[L-1], N>
                - training data outputs.

        Returns:
            : number
    """
    return - np.average(y * np.log(a) + (1 - y) * np.log(1 - a), axis=1)[0]

def binary_cross_entropy_deriv(a, y):
    result = (- np.sum(y / a - (1 - y) / (1 - a), axis=1) / y.shape[1])[0]
    return result


train_x, train_y, test_x, test_y, classes = load_cat_data()

net = BabyNet([12288,7,1])

for c in range(2):
    net.train_batch(train_x, train_y)


    z, a = net.forward_prop(train_x)

    #print(train_y.shape)
    print(a[2])
    #print(binary_cross_entropy(a[2], train_y))

    #totalTrainError = 0
    # totalTrainError += a[2]
    #for testX, testY in zip(standardized_test_x.T, test_y.T):
    #    error = testY[0] - net.forward_prop(testX)
    #    totalTestError += error

    #testAccuracy = totalTestError / len(test_y)
    # trainAccuracy = totalTrainError / len(train_y)
    # print('Loss: ', trainAccuracy)