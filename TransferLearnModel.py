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

class TransferLearnModel(object):

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

    def train_flow(self, dir_path, model_type, num_class, epoch, batch_size=128, lr=0.01, es=True, decay_lr=False,
                   clr=False, tl=True):
        """
        A function to train the data on fly
        :param dir_path: the directory of where the images store
        :param model_type: the type of the model architecture, available options are 'vgg16', 'vgg19', 'resnet50' and 'xception'
        :param num_class: number of classes
        :param epoch: number of epochs to train the model
        :param batch_size: batch size for mini-batch training
        :param lr: learning rate
                - if cyclical learning rate is used, it's the minimum learning rate
                - if step decay learning rate is used, it's the initial learning rate
        :param es: whether to use early stopping or not
        :param decay_lr: whether to use learning rate decay
        :param clr: whether to use cyclical learning rate
        :param tl: whether to use transfer learning(fine-tuning all layers) or not
        :return: the trained model
        """
        # if not apply transfer learning
        if tl == False:
            load_weights = None
        # if apply transfer learning, load weights from ImageNet
        if tl == True:
            load_weights = 'imagenet'

        if model_type.lower() == 'vgg16':
            self.img_shape = (224, 224)
            base_model = VGG16(include_top=False, weights=load_weights, input_shape=(224, 224, 3))
        elif model_type.lower() == 'vgg19':
            self.img_shape = (224, 224)
            base_model = VGG19(include_top=False, weights=load_weights, input_shape=(224, 224, 3))
        elif model_type.lower() == 'resnet50':
            self.img_shape = (224, 224)
            base_model = ResNet50(include_top=False, weights=load_weights, input_shape=(224, 224, 3))
        elif model_type.lower() == 'xception':
            self.img_shape = (299, 299)
            base_model = Xception(include_top=False, weights=load_weights, input_shape=(299, 299, 3))
        else:
            raise ValueError("Error: model name not valid!")

        self.batch_size = batch_size

        x = base_model.output
        # flatten the output
        x = Flatten()(x)

        # the number of units in the dense layer is 1024
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)

        predictions = Dense(num_class, activation="softmax", name='new_dense_layer')(x)
        model = Model(input=base_model.input, output=predictions)
        optimizer = optimizers.SGD(lr=lr, momentum=0.9)
        model.compile(loss="categorical_crossentropy", metrics=["accuracy"],
                      optimizer=optimizer)

        # data augmentation
        # create generator for augmenting training data
        train_datagen = ImageDataGenerator(featurewise_center=True,
                                           zoom_range=0.1,
                                           shear_range=0.6,
                                           rescale=1. / 255,
                                           rotation_range=6
                                           )
        # fit the train_datagen (compute statistics for pre-processing) with some sample training data
        sample_train = self.sample_train(dir_path, 100)
        train_datagen.fit(sample_train)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(dir_path + 'train', target_size=self.img_shape,
                                                            batch_size=batch_size)
        valid_generator = test_datagen.flow_from_directory(dir_path + 'validation', target_size=self.img_shape,
                                                           batch_size=batch_size)

        # Check point: save the model with the best accuracy
        model_path = self.name + '_model.h5'
        check_point = ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True, mode='max')

        callback_list = [check_point]

        # if decay learning rate scheduler is used
        if decay_lr == True:
            lrate = LearningRateScheduler(self.step_decay)
            callback_list.append(lrate)

        # if clr = True, use Cyclical Learning rate
        if clr == True:
            clr_stepsize = 2 * math.ceil(37882 / batch_size)
            clr_triangular = CyclicLR(mode='triangular', base_lr=lr, max_lr=6 * lr, step_size=clr_stepsize)
            callback_list.append(clr_triangular)

        # if es == True, use Early Stoppinp
        if es == True:
            early_stop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
            callback_list.append(early_stop)

        model.fit_generator(train_generator, validation_data=valid_generator, epochs=epoch, callbacks=callback_list)

        # get a map from real label to prediction
        label_map = (train_generator.class_indices)
        # swap value and key, map from prediction to real label
        label_map = dict((v, k) for k, v in label_map.items())
        # store the label map
        with open('label_map.json', 'w') as fp:
            json.dump(label_map, fp)

        return model

    def train(self, X_train, Y_train, X_val, Y_val, base_model, num_class,
                      freeze_num=-1, epoch=1, batch_size=128, lr=0.001):
        """
        A function to train transfer learning model from reading in the whole dataset
        :param base_model: pre-trained model
        :param num_classes: number of classes in the softmax layer
        :param freeze_num: number of layers to freeze, if not specified fine tune all layers
        :param epoch: number of epochs to train the model
        :param batch_size: batch size for mini-batch training
        :param lr: learning rate for use in optimizers
        :return: the final model
        """

        self.batch_size = batch_size

        if freeze_num != -1:
            # freeze the first freeze_num number of layers, fine-tune all the other layers
            for layer in base_model.layers[:freeze_num]:
                layer.trainable = False


        x = base_model.output

        x = Flatten()(x)
        # the number of units in the dense layer is 1024
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_class, activation="softmax", name='new_dense_layer')(x)
        model = Model(input=base_model.input, output=predictions)
        model.compile(loss="categorical_crossentropy", metrics=["accuracy"],
                      optimizer=optimizers.SGD(lr=lr, momentum=0.9))

        # data augmentation
        # create generator for augmenting training data
        train_datagen = ImageDataGenerator(featurewise_center=True,
                                           zoom_range=0.2,
                                           shear_range=0.2,
                                           rescale=1./255,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           rotation_range=40)


        #fit data with train_datagen
        train_datagen.fit(X_train)

        # train model using data augmentation with datagen
        model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size),
                            validation_data=(X_val, Y_val), epochs=epoch)

        # methods to save model: https://stackoverflow.com/a/47271117/8452935
        # save entire model, including architecture, weights, training configuration and state of optimizer
        # can be loaded again to resume training if necessary
        model.save(self.name+'_model.h5')
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


