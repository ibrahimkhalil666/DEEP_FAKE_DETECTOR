#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install kaggle')


# In[8]:


import shutil
import os

# Create the .kaggle directory if it doesn't exist
kaggle_dir = os.path.expanduser("~/.kaggle")
if not os.path.exists(kaggle_dir):
    os.makedirs(kaggle_dir)

# Copy the Kaggle API key file to the .kaggle directory
shutil.copy("C:/Users/Hp/Downloads/kaggle.json", os.path.join(kaggle_dir, "kaggle.json"))

# Set appropriate permissions for the Kaggle API key file
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)


# In[9]:


# Import the Kaggle API client
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate with the Kaggle API using your API key
api = KaggleApi()
api.authenticate()

# List datasets
datasets = api.dataset_list()

# Print the names of the first 10 datasets
for dataset in datasets[:10]:
    print(dataset.ref)


# In[15]:


import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# In[16]:


tf.random.set_seed(42)

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 10


# In[17]:


# Function to load dataset using ImageDataGenerator for batch loading
def load_dataset_batch(data_dir, batch_size, delete_half=False):
    # Create ImageDataGenerator for data augmentation and preprocessing
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    # Generate batches of images and labels from the specified directory
    data_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,  # Resize images to desired size
        batch_size=batch_size,
        class_mode='binary',  # Assuming binary classification (REAL vs FAKE)
        shuffle=True
    )
    
    # Get the number of samples in the dataset
    num_samples = len(data_generator.filenames)
    
    # Get class labels from the data generator
    labels = data_generator.classes
    
    # Convert class labels to binary (REAL: 0, FAKE: 1)
    labels = np.where(labels == 0, 0, 1)
    
    if delete_half:
        # Delete half of the data
        data_generator.samples = int(num_samples / 2)
        num_samples = data_generator.samples
        labels = labels[:num_samples]
    
    return data_generator, num_samples, labels

# Load train, validation, and test datasets using batch loading
train_generator, num_train_samples, train_labels = load_dataset_batch("C:/Users/Hp/Downloads/Dataset/Train", BATCH_SIZE, delete_half=True)
val_generator, num_val_samples, val_labels = load_dataset_batch("C:/Users/Hp/Downloads/Dataset/Validation", BATCH_SIZE)
test_generator, num_test_samples, test_labels = load_dataset_batch("C:/Users/Hp/Downloads/Dataset/Test", BATCH_SIZE)

# Print number of samples in each dataset
print("Number of training samples:", num_train_samples)
print("Number of validation samples:", num_val_samples)
print("Number of test samples:", num_test_samples)


# In[19]:


import matplotlib.pyplot as plt

# Function to display images with labels
def show_images(images, labels, num_images=7):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title('FAKE' if labels[i] == 1 else 'REAL')
        plt.axis('off')
    plt.show()

# Preview a batch of images and labels
batch_images, batch_labels = next(train_generator)
show_images(batch_images, batch_labels)


# In[20]:


import os
import matplotlib.pyplot as plt

def check_class_balance(data_dir):
    classes = os.listdir(data_dir)
    class_counts = {}
    for cls in classes:
        class_counts[cls] = len(os.listdir(os.path.join(data_dir, cls)))
    
    return class_counts

def plot_class_distribution(class_counts):
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.show()

# Specify the directory containing the dataset
data_dir = "C:/Users/Hp/Downloads/Dataset/Train"  # Update with your dataset directory

# Check class balance
class_counts = check_class_balance(data_dir)

# Plot class distribution
plot_class_distribution(class_counts)


# In[21]:


# Build CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile CNN model
cnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


# In[22]:


# Train CNN model
cnn_history = cnn_model.fit(train_generator, 
                            epochs=NUM_EPOCHS, 
                            validation_data=val_generator)


# In[23]:


# Build Xception base model
xception_base = Xception(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Freeze Xception base model layers
for layer in xception_base.layers:
    layer.trainable = False


# In[24]:


# Add custom top layers for Xception model
xception_model = models.Sequential([
    xception_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])


# In[32]:


cnn_model_path = 'cnn_modeldlp.h5'
cnn_model.save(cnn_model_path)
print("Models saved successfully.")


# In[33]:


cnn_model


# In[34]:


# Compile Xception model
xception_model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])


# In[35]:


# Train Xception model
xception_history = xception_model.fit(train_generator, 
                                      epochs=3, 
                                      validation_data=val_generator)


# In[37]:


# Evaluate CNN model on test set
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(test_generator)

# Evaluate Xception model on test set
xception_test_loss, xception_test_acc = xception_model.evaluate(test_generator)

# Plot CNN training history
plt.plot(cnn_history.history['accuracy'], label='CNN Training Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Xception training history
plt.plot(xception_history.history['accuracy'], label='Xception Training Accuracy')
plt.plot(xception_history.history['val_accuracy'], label='Xception Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[38]:


# Print CNN test accuracy
print("CNN Test Accuracy:", cnn_test_acc)

# Print Xception test accuracy
print("Xception Test Accuracy:", xception_test_acc)


# In[39]:


# Define file paths to save the models
cnn_model_path = 'cnn_model.h5'
xception_model_path = 'xception_model.h5'


# Save CNN model
cnn_model.save(cnn_model_path)

# Save Xception model
xception_model.save(xception_model_path)

print("Models saved successfully.")

