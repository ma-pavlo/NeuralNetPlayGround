import numpy as np
import h5py
import matplotlib.pyplot as plt
from main import load_cat_data


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z):
    return z * (1 - z)

def relu(z):
    return np.maximum(0,z)

def d_relu(z):
    return z >= 0 


train_x, train_y, test_x, test_y, classes = load_cat_data()


epochs= 2500 # 1
X = train_x # input data (12288,209)
Y = train_y
learning_rate = 0.001 #0.01
train_error = []
test_error = []

# initialise 
np.random.seed(1)  
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
    #print('A1 shape: ',A1.shape)

    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    #print("A2: ", A2.shape)

    # cost calculation
    cost = np.squeeze((-np.dot(Y,np.log(A2).T) - np.dot(1-Y, np.log(1-A2).T))/ Y.shape[1])

    # backprop
    error = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
    # print(error.shape)

    dZ2 = error * d_sigmoid(A2)
    # print('dZ2 shape: ',dZ2.shape)
    # print('W2 shape: ', W2.shape)
    dA1 = np.dot(W2.T,dZ2)
    # print('dA1 shape: ', dA1.shape)
    dW2 = np.dot(dZ2,A1.T) / A1.shape[1]  # divided by 209 num of examples to average the update
    db2 = np.dot(dZ2, np.ones([dZ2.shape[1],1])) / A1.shape[1] 

    dZ1 = dA1* d_relu(Z1)
    dA0 = np.dot(W1.T,dZ1)
    dW1 = np.dot(dZ1,X.T) / X.shape[1]
    db1 = np.dot(dZ1, np.ones([dZ1.shape[1],1])) / X.shape[1]
    # print(dW1 * learning_rate)

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
    test_cost = np.squeeze((-np.dot(test_y,np.log(A2_t).T) - np.dot(1-test_y, np.log(1-A2_t).T))/test_y.shape[1])

    var += 1
    if (var % 100 == 0):
        train_error.append(cost) 
        test_error.append(test_cost)
        print('Train error: ', str(cost), 'Test error: ', str(test_cost))  #aka train cost/loss 

plt.figure(figsize=(14,4));
plt.plot(train_error);
plt.plot(test_error);


