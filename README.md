
# A voir et à savoir : 

### Les réseaux neuronaux convolutifs (CNN) ont de larges applications dans la reconnaissance d'images et vidéos, les systèmes de recommandations et le traitement du langage naturel.
* How convolutional network work? : https://www.youtube.com/watch?v=FmpDIaiMIeA (by Brandon Rohrer)
* A beginner guide to understand CNN : https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
* Convolutional Neural Networks: https://youtu.be/M7smwHwdOIA (by Yann lecun) 
* Stanford lecture on Convolutional Neural Networks:  https://youtu.be/LxfUGhug-iQ (by Andrej Karpathy)

# How-to-make-a-Tensorflow-Image-Classifier? 
* How to Make a Tensorflow Image Classifier (LIVE): https://youtu.be/APmF6qE3Vjc
* The code for this video https://github.com/llSourcell/How_to_make_a_tensorflow_image_classifier_LIVE/blob/master/demonotes.ipynb

# Les differentes sections du code expliquées: 
>Part 1- Prepare the data 
from tensorflow.examples.tutorials.mnist import input_data
```
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
##The class-labels are One-Hot encoded, which means that each label is a vector with 10 elements
```

>Part 2-Layers parameters  
