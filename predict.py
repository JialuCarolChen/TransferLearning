from pandas import DataFrame
from ImageLoader import ImageLoader
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator




model_path = '/home/ubuntu/Code_AWS/vgg16_fn4_model.h5'
test_dir = ''
dir_path = "/Users/chenjialu/Desktop/DL_Assignment2/Assignment-2-Dataset-Round-1/"

# get validation data
il = ImageLoader(dir_path, model_type='vgg')
X_valid, Y_valid = il.load_valid(6262)
# load model
model = load_model(model_path)

test_datagen = ImageDataGenerator(featurewise_center=True, rescale=1./255)

test_datagen.fit(X_valid)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=1)

predict = model.predict_generator(test_generator,steps = nb_samples)