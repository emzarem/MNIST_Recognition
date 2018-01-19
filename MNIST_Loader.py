"""
MNIST_Loader.py
This is used to load the MNIST data from a .pkl.gz file into three arrays to be used in our neural net.
------------------------------------------------------------------------------------------------------
This code was written while following along with the examples found in Michael Nielsen's online book
http://neuralnetworksanddeeplearning.com/

The purpose of this code was to act as a learning aid as I increase my knowledge of machine learning.

Latest update: January 15 / 2018 - Cleaned up and added test image output.
"""

#standard libraries
import cPickle
import gzip
from PIL import Image

#3rd party libraries
import numpy as np

""" load_data
        This returns tuple of training, validation, and test data.
        Training data is tuple of 50,000 x 784 array of images, and 
        50,000 x 1 array of the corresponding number for each image.
        Validation and test data same but with only 10,000.
"""
def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return(training_data, validation_data, test_data)

""" load_data_wrapper
        Reformatted version of load data where the return values 
        are a 50,000 list of tuples (x,y) where x is 784 array of 
        pixel values and y is the corresponding number label.
"""
def load_data_wrapper():
    tr_d, va_d, te_d  = load_data()

    training_inputs   = [np.reshape(x, (784,1)) for x in tr_d[0]]
    training_results  = [vectorized_result(y) for y in tr_d[1]]
    training_data     = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data   = zip(validation_inputs, va_d[1])

    test_inputs       = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data         = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data)

""" load_custom_image
        This function will load a custom image from a provided file path and manipulate it to 
        conform to the proper shape and properties of the MNIST data.
"""
def load_custom_image(image_path, label):
    given_image = Image.open(image_path)
    given_image = given_image.resize((28,28))
    result = np.asarray(given_image)
    result = np.dot(result[...,:3], [0.299, 0.587, 0.114])
    result = result.flatten()
    result = result.transpose()
    result = np.reshape(result, (784,1))
    for i in xrange(0, len(result)):
        result[i] = [result[i]/255];
    return [[result, label]]

""" vectorized_result
        Returns a 10x1 'one-hot' vector corresponding to j
"""
def vectorized_result(j):
    e    = np.zeros((10, 1))
    e[j] = 1.0

    return e



