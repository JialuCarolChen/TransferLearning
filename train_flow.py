import time
import os
from TransferLearnModel import TransferLearnModel
from DataConvert import convert_dir, create_dir

# Please change the following parameters to run the script:
# The path of the data folder
dir_path = "/Users/chenjialu/Desktop/DL_Assignment2/Assignment-2-Dataset-Round-1/"
# The prefix of the name of the output files
file_prefix = 'resnet50'
# The type of the model architecture, available options are 'vgg16', 'vgg19', 'resnet50' and 'xception'
model_type = 'resnet50'
# Whether to use early stopping or not
use_es = True
# Maximum number of epochs to run
epochs = 100
# Whether to use cyclical learning rate
use_clr = False
# Whether to use drop decay learning rate
use_decay_lr = False
# learning rate:
# if cyclical learning rate is used, it's the minimum learning rate
# if step decay learning rate is used, it's the initial learning rate
lr = 0.01
# batch size
bsize = 64
# Whether to apply transfer learning or not
is_tl = True


# convert the data directory to the structure that fits with the Keras flow_from_directory function
# a new data directory called 'data' is created under the dataset folder
if not os.path.exists(dir_path+'data'):
    convert_dir(dir_path)
# specify the path to the new data directory
dir_path = dir_path+'data/'

# build a class to the train model
tm = TransferLearnModel(file_prefix)

# start recording the training time
st1 = time.time()

# train the model
model=tm.train_flow(dir_path, model_type=model_type, num_class=62, epoch=epochs, batch_size=bsize,
                    lr=lr, es=use_clr, decay_lr=use_decay_lr, clr=use_clr, tl=is_tl)


# record the time
train_time = time.time() - st1
with open(tm.name+"_time.txt", "w") as text_file:
    text_file.write("---Time to train the main model %s seconds ---" % (train_time))

# plot the prediction result
tm.plot_predict(model)

