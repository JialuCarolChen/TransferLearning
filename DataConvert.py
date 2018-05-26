import os
from shutil import copyfile

dir_path = "/Users/chenjialu/Desktop/DL_Assignment2/Assignment-2-Dataset-Round-1/"
# if not os.path.exists(os.pardir.dirname(dir_path+'Data'))

def create_dir(dir_path):
    """
    A function to make directory with structure to work with keras flow_from_directory function
    :param dir_path: the directory path of where the original data store
    """
    dat_path = dir_path+"data"
    if not os.path.exists(dat_path):
        # created a data/ folder if not exist
        os.makedirs(dat_path)
        # created train/ and validation/ subfolders inside data/
        os.makedirs(dat_path+'/train')
        os.makedirs(dat_path+'/validation')
        # created subfolders inside train/ and validation/, each subfolder stores images of a class
        for i in range(62):
            os.makedirs(dat_path + '/train/' + str(i))
            os.makedirs(dat_path + '/validation/' + str(i))

def convert_dir(dir_path):
    """
    A function to convert directory with structure to work with keras flow_from_directory function
    :param dir_path: the directory path of where the original data store
    """
    # make target directories to store the images
    create_dir(dir_path)
    # read training images and convert
    with open(dir_path + 'train.txt', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip('\n').split(' ')
            image = line[0]
            label = line[1]
            src_path = dir_path + 'train-set/' + image
            target_path = dir_path + 'data/train/' + label + "/" + image
            copyfile(src_path, target_path)
            print("copy file to: "+target_path)

    with open(dir_path + 'vali.txt', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip('\n').split(' ')
            image = line[0]
            label = line[1]
            src_path = dir_path + 'vali-set/' + image
            target_path = dir_path + 'data/validation/' + label + "/" + image
            copyfile(src_path, target_path)
            print("copy file to: " + target_path)

convert_dir(dir_path)





