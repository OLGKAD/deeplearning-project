from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model
import keras.datasets.cifar10 as cf10
import keras.preprocessing.image as im
import keras.utils.np_utils as npu
import numpy as np

# Silence debug-info (relevant for tensorflow-gpu)
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Consider splitting some training data into validation data
(x_train, y_train), (x_test, y_test) = cf10.load_data()

# Parameters
NUM_CLASSES = 10
batch_size = 32
updates_per_epoch = len(x_train) / batch_size
epochs = 1

# Add padding to the inner parts of the input to fit the 48x48 minimum requirement
# Using 'mean' strategy
x_train = np.pad(x_train, [(0, 0), (8, 8), (8, 8), (0, 0)], 'mean')
x_test = np.pad(x_test, [(0, 0), (8, 8), (8, 8), (0, 0)], 'mean')

# Turn labels into one-hot encodings
y_train = npu.to_categorical(y_train, NUM_CLASSES)
y_test = npu.to_categorical(y_test, NUM_CLASSES)

# Data batch generator
datagen = im.ImageDataGenerator()

# Computes necessary details for feature normalizations (flip/rotate/shift/etc)
datagen.fit(x_train)


# Create pretrained VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

ext_model = vgg16_model.output
ext_model = Flatten()(ext_model)
ext_model = Dense(1024, activation='relu')(ext_model)
ext_model = Dropout(0.5)(ext_model)
ext_model = Dense(1024, activation='relu')(ext_model)
output = Dense(10, activation='softmax')(ext_model)

model = Model(inputs = vgg16_model.input, outputs = output)

model.compile(
    loss = 'categorical_crossentropy', 
    optimizer = optimizers.SGD(lr = 0.001, momentum = 0.9),
    metrics=['accuracy'])

model.fit_generator(
    datagen.flow(x_train, y_train, batch_size),
    steps_per_epoch = updates_per_epoch,
    epochs = epochs)
del x_train
del y_train
res = model.test_on_batch(x_test, y_test)
print(res)
