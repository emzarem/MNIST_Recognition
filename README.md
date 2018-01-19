# MNIST_Recognition

This code was written while following along with the examples found in Michael Nielsen's online book
http://neuralnetworksanddeeplearning.com/

As such all credit for original Neural Net design and implementation goes to Michael Nielsen.
His works can be found here: 
https://github.com/mnielsen/neural-networks-and-deep-learning

This is my implementation of his MNIST classifier code showcased in his online book and hosted on his Github.

I have added a number of features in my implementation to make it more helpful for learning neural nets, including visualisation
and the ability to use the classifier on your own supplied images.

# Input Options
The input arguments were set up to simplify testing the network:
```
 The following options are available:
     -h (help)     : Will display this output
     -t (train)    : Set 1 or 0 to indicate whether or not you want to train it again.
     -v (visual)   : Set 1 or 0 to indicate whether or not you want a visual test.
     -f (filename) : Specify a file patch after this of a file to test (must train first, set the 
                     name of the file to be the correct number).
```

# Visualisation of Testing
This implementation includes functionality to display test data by showing the determined probability of
it being each possible number:
![alt text](https://raw.githubusercontent.com/emzarem/MNIST_Recognition/master/Examples/test_idx_3538.jpg)




# User Supplied Images
Additionally, this can be used to classify handwritten numbers supplied by the user. Inside MNIST_Loader, load_custom_image() will 
handle converting the image into the same format as the MNIST data by scaling it and changing to grayscale values between 0 and 1.



For example one can draw a number in MS Paint:

![alt text](https://raw.githubusercontent.com/emzarem/MNIST_Recognition/master/Examples/Painting.jpg)

Then save in the format CORRECTLABEL.jpg or .png:

![alt text](https://raw.githubusercontent.com/emzarem/MNIST_Recognition/master/Examples/2.jpg)




And then test it by using '-v 1 -f 2.jpg' :
![alt text](https://raw.githubusercontent.com/emzarem/MNIST_Recognition/master/Examples/test_idx_0.jpg)


