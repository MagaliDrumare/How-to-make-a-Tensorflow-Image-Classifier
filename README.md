
![alt tag](http://parse.ele.tue.nl/cluster/2/CNNArchitecture.jpg)
* Use a CNN to recognize handwritten digits from the MNIST data-set.

# A voir et à savoir : 

### Les réseaux neuronaux convolutifs (CNN) ont de larges applications dans la reconnaissance d'images et vidéos, les systèmes de recommandations et le traitement du langage naturel.
* How convolutional network work? : https://www.youtube.com/watch?v=FmpDIaiMIeA (by Brandon Rohrer)
* A beginner guide to understand CNN : https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
* Convolutional Neural Networks: https://youtu.be/M7smwHwdOIA (by Yann lecun) 
* Stanford lecture on Convolutional Neural Networks: https://youtu.be/AQirPKrAyDg (by Andrej Karpathy)

### Réaliser un classifieur d'images avec tensotflow avec un CNN


* TensorFlow Tutorial #02 Convolutional Neural Network: https://youtu.be/HMcx-zY8JSg
* How to Make a Tensorflow Image Classifier (LIVE): https://youtu.be/APmF6qE3Vjc
* Code for these videos: 
https://github.com/llSourcell/How_to_make_a_tensorflow_image_classifier_LIVE/blob/master/demonotes.ipynb

### Le layer CNN est composé de trois blocs séparés : convolution, reLu, max-pooling 
```python
def new_conv_layer(input,              # The previous layer.
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

# Block 1: Create the TensorFlow operation for convolution. 
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases

# Block 2: Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

# Block 3 : Rectified Linear Unit (ReLU).
    layer = tf.nn.relu(layer)
    return layer, weights
```
### Ces trois blocs peuvent être répétés dans le cadre de la construction du modèle (2 fois)
```python
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
 ```


