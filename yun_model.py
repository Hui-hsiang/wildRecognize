import numpy as np
import keras
import glob
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation,MaxPool2D
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

img_shape = (224, 224, 3)

model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(3,3),input_shape=(224,224,3),strides=(1,1),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

model.add(Conv2D(filters=32,kernel_size=(3,3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(Conv2D(filters=64,kernel_size=(3,3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(Conv2D(filters=128,kernel_size=(3,3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(Flatten())
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(4,activation='softmax'))

model.compile(loss= 'categorical_crossentropy', optimizer = optimizers.Adam(1e-5), metrics=['accuracy'])
model.summary()

images = np.load("images_array.npy")
labels = np.load("labels_array.npy")


from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2)
print(x_train.shape,y_train.shape,x_val.shape,y_val.shape, sep='\n')

batch_size = 32
epochs = 50

train_datagen = ImageDataGenerator(featurewise_center=True, rotation_range=30, zoom_range=0.2, width_shift_range=0.1,
                                   height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(featurewise_center=True)

train_datagen.fit(x_train)
val_datagen.fit(x_val)

train_datagenerator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

save_model_path = 'yun_model.model'

checkpoint = ModelCheckpoint(filepath = save_model_path, monitor='val_loss', save_best_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience = 3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=1)

# class_weight = {0: 50.,
#                 1: 1.}

model.fit_generator(
    train_datagenerator,
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps=x_val.shape[0]//batch_size,
    # class_weight = class_weight,
    callbacks = [checkpoint, reduce_lr, early_stopping]
    )
