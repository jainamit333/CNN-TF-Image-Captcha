import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join, exists

print("will convert image to array/matrix")



def load_data(name_of_data):

    directory = 'images/' + name_of_data
    if exists(directory):

        onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

        for i in range(len(onlyfiles)):
            print(onlyfiles[i])

def convert_image_to_array(path):

    img = Image.open(path).convert('RGBA')
    arr = np.array(img)
    print(arr.shape)
    shape = arr.shape
    flat_arr = arr.ravel()
    vector = np.matrix(flat_arr)
    return vector

def create_label(path):
    return path.split('_')[1].split('.')[0]

#load_data('test')
#create_label('99_2.png')
print(convert_image_to_array('images/test/99_2.png').shape)
