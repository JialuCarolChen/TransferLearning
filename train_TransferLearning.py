from ImageLoader import ImageLoader
from keras.applications import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import time
from TransferLearn import TransLearn

# loading the data
il = ImageLoader("/Users/chenjialu/Desktop/DL_Assignment2/Assignment-2-Dataset-Round-1/", model_type='inception')
X_train, Y_train = il.load_train()
X_valid, Y_valid = il.load_valid()
# one hot encode
Y_train = il.onehot_encode(Y_train)
Y_valid = il.onehot_encode(Y_valid)

print("Finish loading the data...")

# base model
#base_model = VGG16(include_top=False, weights='imagenet', input_shape=X_train.shape[1:])
base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=X_train.shape[1:])

print("Finish loading the base model...")

st1 = time.time()

# Transfer learning on the main data set and get a model
tl = TransLearn("test")
model=tl.fine_tune_fit(X_train, Y_train, X_valid, Y_valid, base_model, num_class=62, freeze_num=41, epoch=1, batch_size=128, lr=0.001)

# record the time
train_time = time.time() - st1

with open(tl.name+"_time.txt", "w") as text_file:
    text_file.write("---Time to train the main model %s seconds ---" % (train_time))

# plot the prediction result
tl.plot_predict(model)

# store prediction result
tl.predict_store(X_valid, Y_valid, model)

