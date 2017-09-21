
# A voir et à savoir : 

### Les réseaux neuronaux convolutifs (CNN) ont de larges applications dans la reconnaissance d'images et vidéos, les systèmes de recommandations et le traitement du langage naturel.
* Convolutional Neural Networks: https://youtu.be/M7smwHwdOIA (by Yann lecun) 
* Stanford lecture on Convolutional Neural Networks:  https://youtu.be/LxfUGhug-iQ (by Andrej Karpathy)
* A beginner guide to understand CNN : https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/

### Les CNN permettent de classer des images dans des catégories 


# How-to-make-a-Tensorflow-Image-Classifier

>Part 1- Prepare the data 
from tensorflow.examples.tutorials.mnist import input_data
```
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
##The class-labels are One-Hot encoded, which means that each label is a vector with 10 elements
```

>Part 2-Layers parameters  
