#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:31:29 2017

@author: magalidrumare
"""


#Image Classifier 
#Model archirecture : 
#INPUT>CONV1>CONV2>FULLCONNECTEDLAYER1>FULLCONNECTEDLAYER2>SOFTMAX>OUTPUT

#import the dependencies 
import tensorflow as tf
import numpy as np



# Part 1- Prepare the data 

##load data 
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

##The class-labels are One-Hot encoded, which means that each label is a vector with 10 elements
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

##class-numbers as integers for the test-set
data.test.cls = np.argmax(data.test.labels, axis=1)

## We know that MNIST images are 28 pixels in each dimension.
img_size = 28

## Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

## Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

## Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

## Number of classes, one class for each of 10 digits.
num_classes = 10


# Part 2-Layers parameters  
## Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

#more filters, featuer map will b
## Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

## Size of the full connected layer 
fc_size = 128   


#Part 3-Initialize randomly the variables 
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    #equivalent to y intercept
    #constant value carried over across matrix math
    return tf.Variable(tf.constant(0.05, shape=[length]))


 #Part 4-Create a convolutional layer function 
 ##CNN have three blocks separated CON>RELU>MAXPOOL 
 

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


    #Part 5-Create a flattening layer function 
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


    #Part 6 - Create a full connected layer function  

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


    #Part 7-Placeholder Variables 

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    #The convolutional layers expect x to be encoded as a 4-dim tensor so we have to reshape it 
    #so its shape is instead [num_images, img_height, img_width, num_channels]
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
	# the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
	#a placeholder variable for the class-number
y_true_cls = tf.argmax(y_true, dimension=1)

	
	#Part 8- Create the model 
	##Convolutional layers 1 and 2 using the convolutional layer function new_conv_layer 

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)


    ## Obtain layer_flat and num_features using the flattening layer function 
layer_flat, num_features = flatten_layer(layer_conv2) #images which have been flattened to vectors of length 1764 each. Note that 1764 = 7 x 7 x 36.

	## Add two full connected layer using the full connected layer function 
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

	##Classification of the image 
	# Prediction using sofmax function that normalize the scores into probabilities 
y_pred = tf.nn.softmax(layer_fc2)
	#The class-number is the index of the largest element.
y_pred_cls = tf.argmax(y_pred, dimension=1)


	#Part 9 - Optimization of the cost function and Accuracy 
	#need to know how well the model currently performs by comparing the predicted output of the model y_pred to the desired output y_true.
	#The cross-entropy is a performance measure used in classification.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
	#we simply take the average of the cross-entropy for all the image classifications.
cost = tf.reduce_mean(cross_entropy)

	#the cost measure that must be minimizedusing an advanced gradient descent form : the Adam Optimizer 
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

	#This is a vector of booleans whether the predicted class equals the true class of each image.
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

	#This calculates the classification accuracy by first type-casting the vector of booleans to floats, so that False becomes 0 and True becomes 1, and then calculating the average of these numbers.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#Part 10 - Training of the model 
	#create a tensorflow session 
session = tf.Session()
	#initialize variables 
session.run(tf.global_variables_initializer())

	#There are 55,000 images in the training-set. It takes a long time to calculate the gradient of the model using all these images. 
	#We therefore only use a small batch of images in each iteration of the optimizer.
train_batch_size = 64

	#Function for performing a number of optimization iterations so as to gradually improve the variables of the network layers. 
	#In each iteration, a new batch of data is selected from the training-set and then TensorFlow executes the optimizer using those training samples. 
	#The progress is printed every 100 iterations.

	

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

   
    

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    