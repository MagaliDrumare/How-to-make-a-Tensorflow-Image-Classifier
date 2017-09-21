
* 

# How-to-make-a-Tensorflow-Image-Classifier

>Part 1- Prepare the data 
from tensorflow.examples.tutorials.mnist import input_data
```
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
##The class-labels are One-Hot encoded, which means that each label is a vector with 10 elements
```

>Part 2-Layers parameters  
