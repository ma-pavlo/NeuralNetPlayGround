import numpy as np
import h5py
import matplotlib.pyplot as plt


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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z):
    return z * (1 - z)

def relu(z):
    return np.maximum(0,z)

def d_relu(z):
    return z >= 0 


train_x, train_y, test_x, test_y, classes = load_cat_data()


epochs=1#2500
X = train_x # input data (12288,209)
Y = train_y
learning_rate = 0.01
train_error = []
test_error = []

# initialise 
np.random.seed(1)   # kann doch nicht das problem von rand, initialisation sein
W1 = np.random.randn(8, 12288) * 0.01
b1 = np.zeros((8, 1))
W2 = np.random.randn(1, 8) * 0.01
b2 = np.zeros((1, 1))
#print(W1.shape, b1.shape)
#print(W2.shape, b2.shape)
    
var = 0
for i in range(epochs):
    # forwardprop
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    print(A1.shape)

    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    #print(A2.shape)

    # cost calculation
    #print(Y.shape)
    #print(A2.shape)
                # yy = np.array([[1, 1, 0, 0]])
                # aa2 = np.array([[1, 2, 3, 1]])
    cost = np.squeeze((-np.dot(Y,np.log(A2).T) - np.dot(1-Y, np.log(1-A2).T)) / Y.shape[1])
    print(cost)
    #train_error.append(cost)

    # backprop
    error = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
    print(error.shape)

    print("A2: ", A2.shape)
    print("sigderiv(Z2): ", d_sigmoid(Z2).shape)
    dZ2 = error * d_sigmoid(A2)
    #print(dZ2.shape)
    #print(dZ2.shape)
    print(W2.shape)
    dA1 = np.dot(W2.T,dZ2)
    print(dA1.shape)
    dW2 = np.dot(dZ2,A1.T) / A1.shape[1]           # wieso teilen wir hier bzw 1/A.shape * was wir haben
    db2 = np.dot(dZ2, np.ones([dZ2.shape[1],1])) / A1.shape[1] 

    dZ1 = dA1* d_relu(Z1)
    dA0 = np.dot(W1.T,dZ1)
    dW1 = np.dot(dZ1,X.T) / X.shape[1]
    db1 = np.dot(dZ1, np.ones([dZ1.shape[1],1])) / X.shape[1]
    #print(dW1 * learning_rate)

    W1 -= dW1 * learning_rate
    W2 -= dW2 * learning_rate
    b1 -= db1 * learning_rate
    b2 -= db2 * learning_rate

    # test
    Z1_t = W1.dot(test_x) + b1
    A1_t = relu(Z1_t)
    Z2_t = W2.dot(A1_t) + b2
    A2_t = sigmoid(Z2_t)

    # cost calculation
    test_cost = np.squeeze((-np.dot(test_y,np.log(A2_t).T) - np.dot(1-test_y, np.log(1-A2_t).T)) / test_y.shape[1])
    #test_error.append(test_cost)

    var += 1
    if (var % 100 == 0):
        train_error.append(cost)
        test_error.append(test_cost)
        print('Train l: ', str(cost), 'Test l: ', str(test_cost))
        #print(error)

#plt.figure(figsize=(14,4));
#plt.plot(train_error);
#plt.plot(test_error);









# y_train = train_y
# #Weights
# w0 = 2*np.random.random((8, 12288)) - 1 #for input   - 4 inputs, 3 outputs
# w1 = 2*np.random.random((1, 8)) - 1 #for layer 1 - 5 inputs, 3 outputs

# #learning rate
# n = 0.01

# #Errors - for graph later
# errors = []

# #Train
# for i in range(100000):

#     #Feed forward
#     layer0 = train_x
#     layer1 = sigmoid(np.dot(w0,layer0))
#     #print(layer1.shape)
#     layer2 = sigmoid(np.dot(w1,layer1))
#     #print(layer2.shape)

#     #Back propagation using gradient descent
#     layer2_error = y_train - layer2
#     layer2_delta = layer2_error * sigmoid_deriv(layer2)
    
#     layer1_error = np.dot(w1.T,layer2_delta)
#     layer1_delta = layer1_error * sigmoid_deriv(layer1)
    
#     w1 += np.dot(layer2_delta,layer1.T) * n
#     w0 += np.dot(layer1_delta,layer0.T) * n
    
#     error = np.mean(np.abs(layer2_error))
#     errors.append(error)
#     accuracy = (1 - error) * 100
#     print(error)

# #Plot the accuracy chart
# plt.plot(errors)
# plt.xlabel('Training')
# plt.ylabel('Error')
# plt.show()
        
# print("Training Accuracy " + str(round(accuracy,2)) + "%")