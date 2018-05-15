import keras.datasets.cifar10 as cf10
import keras.preprocessing.image as im
import keras.utils.np_utils as npu
import numpy as np
import os
import os.path
import sys
import tensorflow as tf
import util

# Weights file path
weight_file = 'weights/vgg16_transfer.h5'

# Silence debug-info (relevant for tensorflow-gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

# Exit early if no mode is given
if len(sys.argv) < 2:
    print('A mode needs to be specified')
    print('Usage: python main.py [train|evaluate]')
    exit()

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

# Fetch the model (or create it if there is none)
model = util.get_model()

# Data batch generator
datagen = im.ImageDataGenerator()

# Tensorflow with GPU enabled easily runs out of memory. 
# So we need to train and evaluate seperately
mode = os.sys.argv[1]
if mode == 'train':

    # Computes necessary details for feature normalizations (flip/rotate/shift/etc)
    datagen.fit(x_train)

    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size),
        steps_per_epoch = updates_per_epoch,
        epochs = epochs)
    util.save_model(model)
elif mode == 'evaluate':
    for i in range(100):
        x_batch = x_test[i*10:(i+1)*10]
        y_batch = y_test[i*10:(i+1)*10]
        res = model.test_on_batch(x_batch, y_batch)
        print(res)
