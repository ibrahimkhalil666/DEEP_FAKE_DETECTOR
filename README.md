# DEEP_FAKE_DETECTOR

##Deep Fake Detector
#Overview
The Deep Fake Detector project aims to develop a deep learning solution for detecting deep fake images. This repository contains the code for training and evaluating two models: a Convolutional Neural Network (CNN) and an Xception-based model.

#Features
Train a CNN model from scratch for deep fake detection.
Utilize transfer learning with the Xception architecture for improved performance.
Preprocess and augment datasets using TensorFlow's ImageDataGenerator.
Visualize training histories and evaluate models on test datasets.


#Usage
Prepare your dataset:
Organize your dataset into train, validation, and test directories.
Ensure each directory contains subdirectories for each class (e.g., real and fake).
Update the dataset paths:
Modify the paths in the code to point to your dataset directories.
Train and evaluate models:
Run the provided scripts to train and evaluate the CNN and Xception models.
