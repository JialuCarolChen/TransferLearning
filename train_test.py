from ImageLoader import ImageLoader
from keras.applications import VGG16
from keras.applications import resnet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import time
from TransferLearn import TransLearn

# path of the data folder
dir_path = "/home/ubuntu/Assignment-2-Dataset-Round-1/data/"

# base model 1: VGG 16, freezing the first block (3 layers, freeze_num=4)
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# base model 2: RestNet50
# base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=X_train.shape[1:])

# bast model 3: Inception ResNet V2, freeze the stem block (the first 40 layers), freeze_num=41
# base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=X_train.shape[1:])

print("Finish loading the base model...")

st1 = time.time()

# Transfer learning on the main data set and get a model
tl = TransLearn("vgg16_fn4")
model=tl.fine_tune_fit_flow(dir_path, base_model, model_type='vgg', num_class=62, freeze_num=4, epoch=1, batch_size=128, lr=0.001)

# record the time
train_time = time.time() - st1

with open(tl.name+"_time.txt", "w") as text_file:
    text_file.write("---Time to train the main model %s seconds ---" % (train_time))

# plot the prediction result
tl.plot_predict(model)

# get validation data
il = ImageLoader("/home/ubuntu/Assignment-2-Dataset-Round-1/", model_type='inception')
X_valid, Y_valid = il.load_valid(6262)
Y_valid = il.onehot_encode(Y_valid)
# store prediction result
tl.predict_store(X_valid, Y_valid, model)

