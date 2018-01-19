"""
Network.py
This implements a simple neural network with numpy.
Stochastic gradient descent is used.
-----------------------------------------------------------------------------------------------------
This code was written while following along with the examples found in Michael Nielsen's online book
http://neuralnetworksanddeeplearning.com/

The purpose of this code was to act as a learning aid as I increase my knowledge of machine learning.

Latest update: January 15 / 2018 - Cleaned up and added test image output.
"""

import numpy as np
import random as rand
import os
import matplotlib.pyplot as plt


# Extra functions
""" Sigmoid Function """
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

""" Derivative of Sigmoid Function """
def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))


# Network
""" Neural Network class, 
        intakes list of sizes of each layer e.g. [2 3 1]
        indicating 2 nodes in layer 1, 3 in layer 2, 1 in 
        layer 3
"""
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes      = sizes
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases  = [np.random.randn(y,1) for y in sizes[1:]]


   #""" Function to determine output based on input a """
    def feed_forward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    """ Visual Test
            This function will use the already-generated biases and weights to randomly
            test some of the MNIST data. It will output a visual of the image tested along
            with a score for each possible value.
    """
    def visual_test(self, test_data, index=-1):
        try:
            self.weights = np.load('weights.npy')
            self.biases  = np.load('biases.npy')
        except IOError:
            print "No previous training data found."
            return 0

        # Choose random image from provided data if not specified
        if(index<0 or index >= len(test_data)):
            index = rand.randint(0, len(test_data) - 1)

        # Get image
        test_image = test_data[index]

        # Add the test data (generated probability for each number)
        results = self.feed_forward(test_image[0])
        label   = test_image[1]

        # Add to plot
        reshaped_image = np.array(test_image[0]).reshape(28, 28)
        plt.imshow(np.column_stack((reshaped_image, np.ones((28,20)))), cmap = 'gray')
        plt.title('Testing Index ' + str(index))

        # Add test metrics
        for val, i in zip(results, xrange(0, len(results))):
            if(i == np.argmax(results)):
                plt.text(28, 2 + i * 2, str(i) + ': ' + str(val), color='red')
            else:
                plt.text(28, 2 + i * 2, str(i) + ': ' + str(val), color='black')

        plt.text(32, 5 + 2*len(results), 'True value: ' + str(label), color='blue')

        # Finally show the plot and save
        if not os.path.isdir('TestImages'):
            os.makedirs('TestImages')

        plt.savefig('TestImages/test_idx_' + str(index) + '.png')
        plt.show()

        return 1

    """ Mini-Batch Stochastic Gradient Descent Function
            training_data   - list of tuples (input, desired output)
            epochs          - number of passes
            mini_batch_size - size of mini batch for algorithm
            eta             - learning rate
            test_data       - optional test data log
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)

        n = len(training_data)

        # iterate through each epoch
        for j in xrange(epochs):
            np.random.shuffle(training_data)

            # take the mini batches from the training data
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            # apply gradient descent to each mini_batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # log
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

        np.save('weights.npy', self.weights)
        np.save('biases.npy', self.biases)

    """ Gradient Descent Function for Batch  """
    def update_mini_batch(self, mini_batch, eta):
        # init gradient matrices
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            # invoke back propogation
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]

        # update weights and biases
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]

    """ Back propagation function
            Returns (nabla_b, nabla_w) representing gradient for the cost fcn, where 
            nabla_b, nabla_w are lists of a similar nature to biases and weights
    """
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # now feed forward
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # now a backwards pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # L = 1 is last layer, L=2 is second last etc...
        for l in xrange(2, self.num_layers):
            z     = zs[-l]
            sp    = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    """ Evaluation function
            This function returns the number of inputs correctly classified
            by the network.
    """
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]

        return sum(int(x == y) for (x,y) in test_results)

    """ Cost derivative
            Returns vector of d C_x / d a
    """
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

