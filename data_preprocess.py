import numpy as np
import keras
import glob
from keras.utils import to_categorical

from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import itertools
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def centering_image(img):
    size = [256,256]
    
    img_size = img.shape[:2]
    # centering
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img

    return resized

def GetImg(image_path):
    image = cv2.imread(image_path)
    return image
def GetLabel(image_path):
    label_path = image_path.replace("JPG","txt")
    with open(label_path,'rb') as f:
        label = f.readlines()[0]
        label = int(label)
        #print(label)
    return label


def data_preprocessing(image, label):
    if(image.shape[0] > image.shape[1]):
            tile_size = (int(image.shape[1]*256/image.shape[0]),256)
    else:
            tile_size = (256, int(image.shape[0]*256/image.shape[1]))

    #centering
    image = centering_image(cv2.resize(image, dsize=tile_size))
        
    #out put 224*224px 
    image = image[16:240, 16:240]
    label = to_categorical(label, num_classes = 4)
    return image, label

images = []
labels = []

train_images_path = './*/*.JPG'
imagesList = glob.glob(train_images_path)
np.random.shuffle(imagesList)




for img in  imagesList:
    #print (img)
    image = GetImg(img)
    label = GetLabel(img)
    image , label = data_preprocessing(image,label)
    images.append(image)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

images = images.astype('float32')
images /= 255


print (images.shape)
print (labels.shape)
np.save('images_array',images)
np.save('labels_array',labels)



