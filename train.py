from ImageLoader import ImageLoader
from keras.applications import VGG16, resnet50, inception_resnet_v2
from keras.applications import resnet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import time
from TrainModel import TrainModel

# path of the data folder
dir_path = "/home/ubuntu/Assignment-2-Dataset-Round-1/data/"

st1 = time.time()

# Transfer learning on the main data set and get a model
tm = TrainModel("restnet50")
model=tm.train_flow(dir_path, model_type='restnet50', num_class=62, epoch=90, batch_size=64, lr=0.001, clr=True)

# record the time
train_time = time.time() - st1

with open(tm.name+"_time.txt", "w") as text_file:
    text_file.write("---Time to train the main model %s seconds ---" % (train_time))

# plot the prediction result
tm.plot_predict(model)

