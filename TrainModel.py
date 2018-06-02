from keras import optimizers
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import VGG16, VGG19, ResNet50, Xception
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from pandas import DataFrame
from clr_callback import CyclicLR
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import os

class TrainModel(object):

    def __init__(self, name):
        """
        A class to train a single model
        :param name: name of the model (for specify file name)
        """
        self.name = name


    def sample_train(self, dir_path, sample_size):
        """A function to create sample data to fit train_datagen

        :param dir_path:
        :param sample_size: the sample size for each class

        """
        sample_train = np.zeros((62*sample_size, self.img_shape[0], self.img_shape[1], 3))
        row_index = 0
        for i in range(62):
            path = dir_path + 'train/' + str(i) + '/'
            j=0
            for root, dirs, files in os.walk(path):
                for f in files:
                    j+=1
                    if j > sample_size:
                        break
                    # record sample image
                    img_path = path + f
                    # turn image to numpy array
                    img = load_img(img_path)  # this is a PIL image
                    img = img_to_array(img)
                    # print(img.shape)
                    # convert to integer
                    img = img.astype(int)
                    # resize to fit in pre-trained model
                    img = resize(img, self.img_shape)
                    sample_train[row_index, :] = img
                    row_index = row_index+1
        print("Sample training data to fit ImageDataGenerator: " +str(sample_train.shape))
        return sample_train



    def train_flow(self, dir_path, model_type, num_class, epoch = 90, batch_size=128, lr=0.001, clr = True):
        """
        A function to train transfer learning model from reading in data on fly (flow from directory)
        :param dir_path: the directory of where the image store
        :param base_model: pre-trained model
        :param model_type: type of model, available option: restnet50, vgg16, vgg19, Xception
        :param num_class: number of classes in the softmax layer
        :param freeze_num: number of layers to freeze, if not specified fine tune all layers
        :param epoch: number of epochs to train the model, if epoch = -1, use early stopping
        :param batch_size: batch size for mini-batch training
        :param lr: learning rate for use in optimizers
        :return: the final model
        """
        # specify image size
        # input image shape for inception and inception resnet model

        if model_type.lower() == 'vgg16':
            self.img_shape = (224, 224)
            base_model = VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
        elif model_type.lower() == 'vgg19':
            self.img_shape = (224, 224)
            base_model = VGG19(include_top=False, weights=None, input_shape=(224, 224, 3))
        elif model_type.lower() == 'restnet50':
            print("hellp")
            self.img_shape = (224, 224)
            base_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))
        elif model_type.lower() == 'xception':
            self.img_shape = (299, 299)
            base_model = Xception(include_top=False, weights=None, input_shape=(299, 299, 3))
        else:
            raise ValueError("Error: model name not valid!")

        self.batch_size = batch_size

        x = base_model.output

        x = Flatten()(x)
        # the number of units in the dense layer is 1024
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        predictions = Dense(num_class, activation="softmax", name='new_dense_layer')(x)
        model = Model(input=base_model.input, output=predictions)
        model.compile(loss="categorical_crossentropy", metrics=["accuracy"],
                      optimizer=optimizers.Adam())

        # data augmentation
        # create generator for augmenting training data
        train_datagen = ImageDataGenerator(featurewise_center=True,
                                           zoom_range=0.2,
                                           shear_range=0.2,
                                           rescale=1./255,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           rotation_range=30)
        # fit the train_datagen (compute statistics for pre-processing) with some sample training data
        sample_train = self.sample_train(dir_path, 100)
        train_datagen.fit(sample_train)

        test_datagen = ImageDataGenerator(featurewise_center=True, rescale=1./255)
        test_datagen.fit(sample_train)

        train_generator = train_datagen.flow_from_directory(dir_path+'train', target_size=self.img_shape, batch_size=batch_size)
        valid_generator = test_datagen.flow_from_directory(dir_path+'validation', target_size=self.img_shape, batch_size=batch_size)


        # Check point: save the model with the best accuracy
        model_path = self.name + '_model.h5'
        check_point = ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True, mode = 'max')

        callback_list = [check_point]

        # if clr = True, use Cyclical Learning rate
        if clr == True:
            clr_stepsize = 2 * math.ceil(37882 / batch_size)
            clr_triangular = CyclicLR(mode='triangular', base_lr=lr, max_lr=6*lr, step_size=clr_stepsize)
            callback_list.append(clr_triangular)

        # use Early Stoppinp
        early_stop = EarlyStopping(monitor='val_acc', patience=8, mode='max')
        callback_list.append(early_stop)
        model.fit_generator(train_generator, validation_data=valid_generator, epochs=epoch, callbacks=callback_list)

        # methods to save model: https://stackoverflow.com/a/47271117/8452935
        # save entire model, including architecture, weights, training configuration and state of optimizer
        # can be loaded again to resume training if necessary
        # model.save(self.name+'_model.h5')

        # get a map from real label to prediction
        label_map = (train_generator.class_indices)
        # swap value and key, map from prediction to real label
        label_map = dict((v, k) for k, v in label_map.items())
        # store the label map
        with open('label_map.json', 'w') as fp:
            json.dump(label_map, fp)

        return model


    def plot_predict(self, model):
        """
        :param model: final model
        :param prefix: prefix of the figure name
        :return:
        """
        # visualise accuracy and loss, and save figures to disk
        ax = plt.subplots(dpi=200)
        plt.plot(model.history.epoch, model.history.history['acc'], label='Training')
        plt.plot(model.history.epoch, model.history.history['val_acc'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(frameon=False)
        plt.savefig(self.name+'_accuracy.png')

        ax = plt.subplots(dpi=200)
        plt.plot(model.history.epoch, model.history.history['loss'], label='Training')
        plt.plot(model.history.epoch, model.history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(frameon=False)
        plt.savefig(self.name+'_loss.png')

    def predict_store(self, X_dat, Y_dat, model, batch_size=None):
        """
        A function to predict and store the prediction result of the data
        :param X_dat: training data to predict
        :param Y_dat: the labelling data (one-hot-encode format)
        :param batch_size: batch size for prediction (should be same as for training)
        :param model: the model used to predict
        """

        # if batch size note specified, use the same batch size used for training the model
        if batch_size == None:
            batch_size = self.batch_size

        # get the prediction probabilites
        predictions = model.predict(X_dat, batch_size=batch_size)
        # get the gold label
        gold = Y_dat.argmax(axis=1)
        # get the predicted label
        pred = predictions.argmax(axis=1)
        # output csv
        out = DataFrame(predictions)
        out["pred"] = pred
        out["gold"] = gold
        out.to_csv(self.name + "_predictions.csv", header=True)


