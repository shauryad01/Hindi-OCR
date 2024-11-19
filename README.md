# Hindi OCR with k-NN Classifier
This repository provides a simple implementation of Hindi Optical Character Recognition (OCR) using the k-Nearest Neighbors (k-NN) algorithm. The project works with a dataset of handwritten Devanagari numerical characters and can predict the digit in a custom image. This project implements a k-Nearest Neighbors (k-NN) image classification model, using a custom dataset of 28x28 pixel grayscale images, similar to the MNIST dataset. The goal is to predict the class of an image based on the nearest neighbors in the training set.

## Overview
Dataset: The project works with an MNIST-like dataset stored in the data/Dataset/ directory.
Task: Classifying images from a custom dataset using the k-NN algorithm.
Custom Image Prediction: The model can also predict the label of a custom image (test.png) stored in the test/ directory.
## Requirements
Python 3.x
numpy (for numerical operations)
PIL (Python Imaging Library) for image processing
You can install the required dependencies by running:

## File Structure
```
├── data
│   ├── devanagari-test-images-idx3-ubyte      # Test images file
│   ├── devanagari-test-labels-idx1-ubyte      # Test labels file
│   ├── devanagari-train-images-idx3-ubyte     # Training images file
│   ├── devanagari-train-labels-idx1-ubyte     # Training labels file
├── test
│   └── test.png                        # Custom image input to detect 
├── ocr.py                              # The main script for k-NN classification
└── README.md                           # Project documentation
```

## Custom Image Prediction
When DEBUG is set to False, the script will attempt to predict the label for a custom image (test.png). Make sure the image is available in the test/ directory.

## Result
The accuracy of the k-NN classification model is printed at the end of the script, along with the predicted labels for the test dataset.

## Customization
You can adjust the following parameters in the main() function: 
n_train: Number of training images to use. 
n_test: Number of test images to classify. 
k: Number of neighbors to consider in the k-NN algorithm.
