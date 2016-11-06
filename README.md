# MLEND Capstone
Mike Cassell - mcasse01@gmail.com

## Introduction
This is my capstone project for Udacity's ML Engineer Nanodegree. In the project I build and optimized a convolutional nerual network to predict images in the CIFAR 10 dataset. This project was a ton of fun and taught me an increadible amount about convolutional networks, TensorFlow and getting to understand and apply batch normalization and Fractional Max Pooling was great.

## Requirements:
TensorFlow rc 0.11
Numpy
Pandas 
TFLearn
SKLearn (only for the confusion matrix)

## Environment
This solution was developed on Ubuntu 16.04 using an NVIDIA GEForce 750M GPU. Tensorflow was compiled from code to allow for the use of the GPU in training and testing.

## Setup:
Since Kaggle doesn't allow for automated downloading of source files, the testing and training files were unzipped into corresponding test and train folders under the Data directory. The fles for the X comparison were downloaded from the CIFAR-10 website and the test_batch file was unzipped into the Ver folder. Random images from Google can be added to the Web folder for testing against non-CIFAR images as desired (the script will resize and convert them.)

To download the Kaggle dataset, simply login to Kaggle and go to https://www.kaggle.com/c/cifar-10 and select download. For the original CIFAR-10 dataset (used here for generating the Confusion Matrix) you can go to https://www.cs.toronto.edu/~kriz/cifar.html.

## Notes
The final model can be restored from the model.tflearn located in the Final directory. Due to the change to TF rc11, the mdoel generates some warnings on intialization but do not effect the training or inference operations (and should go away in a future TFLearn update.)

## Thanks
I want to thank my wife Kaitlyn for her incredible support and patience in allowing me the time to work on and finish this project. I'd also like to give thanks to the Udacity team, my classmates advice in forums, the TensorFlow team and the TFLearn contributors.
