import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from collections import defaultdict

class ImageLoader:
    def __init__(self, file_path, model_type):
        """
        :param file_path:
        :param model_type:
        """
        self.file_path = file_path

        # specify image size
        # input image shape for inception and inception resnet model
        if model_type.lower() == 'vgg' or model_type.lower() == 'restnet':
            self.img_shape = (224, 224)
        # input image shape for inception and inception resnet model
        elif model_type.lower() == 'inception':
            self.img_shape = (299, 299)
        else:
            raise ValueError("Error: pretrain model name not valid!")

    def load_train(self, n):
        # record training data
        X_train = np.zeros((n, self.img_shape[0], self.img_shape[1], 3))
        Y_train = np.zeros((n,))
        # read in training data
        with open(self.file_path+'train.txt', 'r') as f:
            for i, line in enumerate(f):
                line = line.strip('\n').split(' ')
                # print(i, line)
                image = line[0]
                Y_train[i] = int(line[1])
                img_path = self.file_path + 'train-set/' + image
                # turn image to numpy array
                img = load_img(img_path)  # this is a PIL image
                img = img_to_array(img)
                # print(img.shape)
                # convert to integer
                img = img.astype(int)
                # resize to fit in pre-trained model
                img = resize(img, self.img_shape)
                X_train[i,:] = img


        print("Read in training data with dimensions: " + str(X_train.shape))
        print("Read in training labels with dimensions: "+ str(Y_train.shape))
        return X_train, Y_train


    def load_valid(self, n):
        # record validation data
        X_train = np.zeros((n, self.img_shape[0], self.img_shape[1], 3))
        Y_train = np.zeros((n,))
        # read in validation data
        with open(self.file_path + 'vali.txt', 'r') as f:
            for i, line in enumerate(f):
                line = line.strip('\n').split(' ')
                # print(line)
                image = line[0]
                Y_train[i] = int(line[1])
                img_path = self.file_path + 'vali-set/' + image
                # turn image to numpy array
                img = load_img(img_path)  # this is a PIL image
                img = img_to_array(img)
                # convert to integer
                img = img.astype(int)
                # resize to fit in pre-trained model
                img = resize(img, self.img_shape)
                X_train[i, :] = img

        print("Read in validating/testing data with dimensions: " + str(X_train.shape))
        print("Read in validating/testing labels with dimensions: " + str(Y_train.shape))
        return X_train, Y_train

    def onehot_encode(self, Y):
        return pd.get_dummies(Y).values



#il = ImageLoader("/Users/chenjialu/Desktop/DL_Assignment2/Assignment-2-Dataset-Round-1/", model_type='inception')
#X_train, Y_train = il.load_train(37882)
#X_valid, Y_valid = il.load_valid(6262)
#Y_train = il.onehot_encode(Y_train)
#Y_valid = il.onehot_encode(Y_valid)



"""
def make_batch(files, labels, batchsize, img_dim):
    files,labels = shuffle(files,labels)
    while 1:
        for i in range(0, len(files), batchsize):
            batchX = files[i:i+batchsize]
            batchY = np.array(labels[i:i+batchsize][:])
            batchX = np.array([parse_image(Image.open(fname), img_dim)/255 for fname in batchX])
            yield(batchX, batchY)

"""
