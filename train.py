from ImageLoader import ImageLoader
from keras.applications import VGG16
from keras.applications import resnet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2

import time
from TransferLearnModel import TransferLearnModel

# loading the data
path = "/put/your/path/here"
# number of training data entries
nt = 37882
# number of validation data entries
nv = 6262

il = ImageLoader(path, model_type='inception')
X_train, Y_train = il.load_train(nt)
X_valid, Y_valid = il.load_valid(nv)

# one hot encode
Y_train = il.onehot_encode(Y_train)
Y_valid = il.onehot_encode(Y_valid)

print("Finish loading the data...")

# base model 1: VGG 16, freezing the first block (3 layers, freeze_num=4)
base_model = VGG16(include_top=False, weights='imagenet', input_shape=X_train.shape[1:])

# base model 2: RestNet50
# base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=X_train.shape[1:])

# bast model 3: Inception ResNet V2, freeze the stem block (the first 40 layers), freeze_num=41
# base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=X_train.shape[1:])

print("Finish loading the base model...")

st1 = time.time()

# Transfer learning on the main data set and get a model
tl = TransferLearnModel("vgg16_fn4")
model=tl.train(X_train, Y_train, X_valid, Y_valid, base_model, num_class=62, freeze_num=4, epoch=100, batch_size=128, lr=0.001)

# record the time
train_time = time.time() - st1

with open(tl.name+"_time.txt", "w") as text_file:
    text_file.write("---Time to train the main model %s seconds ---" % (train_time))

# plot the prediction result
tl.plot_predict(model)

# store prediction result
tl.predict_store(X_valid, Y_valid, model)

