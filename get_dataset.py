###
### get_dataset.py
###
### Description:
### This is a script that is part of the dataset that process each image into grayscale images
### and separates them to training and test sets.
###
### This script has been modified to accommodate for validation and test sets.
###

# Arda Mavi
import os
from PIL import Image
import imageio
import numpy as np
from os import listdir
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Settings:
img_size = 64
grayscale_images = True
num_class = 10
test_size = 0.2

"""
def get_img(data_path):
    # Getting image array from path:
    img = imageio.imread(data_path, flatten=grayscale_images)
    img = Image.imresize(img, (img_size, img_size, 1 if grayscale_images else 3))
    return img
"""
    
def get_img(data_path):
    # Getting image array from path:
    img = imageio.imread(data_path, as_gray=grayscale_images)
    
    # Convert to PIL Image
    img = Image.fromarray(img)
    
    # If grayscale, ensure single channel
    if grayscale_images:
        img = img.convert('L')
    
    # Resize the image
    img = img.resize((img_size, img_size))
    
    # Convert back to numpy array
    img = np.array(img)
    
    # If grayscale, add channel dimension
    if grayscale_images:
        img = np.expand_dims(img, axis=-1)
    
    return img

def get_dataset(dataset_path='dataset'):
    # Getting all data from data path:
    try:
        X = np.load('dataset/X.npy')
        Y = np.load('dataset/Y.npy')
    except:
        labels = listdir(dataset_path) # Geting labels
        X = []
        Y = []
        for i, label in enumerate(labels):
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = get_img(datas_path+'/'+data)
                X.append(img)
                Y.append(i)
        # Create dateset:
        X = 1-np.array(X).astype('float32')/255.
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_class)
        print(X.shape)
        print(Y.shape)
        if not os.path.exists('npy_dataset/'):
            os.makedirs('npy_dataset/')
        np.save('npy_dataset/X.npy', X)
        np.save('npy_dataset/Y.npy', Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X_train, X_test, Y_train, Y_test
