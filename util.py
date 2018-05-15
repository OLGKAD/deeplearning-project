from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
import numpy as np
import os.path

extended_vgg16_model_path = "models/extended_vgg16"

def save_model(model):
    model.save(extended_vgg16_model_path)
    print("Saved model successfully!")

def get_model ():
    if (os.path.isfile(extended_vgg16_model_path)):
        print("Found existing model")
        return load_model(extended_vgg16_model_path)
    else:
        print("Could not find existing model... creating extended model from scratch")
        return create_extended_model()

def create_extended_model ():
    # Create pretrained VGG16 model
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    # Freeze the model - We don't want to train it
    for layer in vgg16_model.layers:
        layer.trainable = False

    # Extend the model for transfer learning
    ext_model = vgg16_model.output
    ext_model = Flatten()(ext_model)
    ext_model = Dense(1024, activation='relu')(ext_model)
    ext_model = Dropout(0.5)(ext_model)
    ext_model = Dense(1024, activation='relu')(ext_model)
    output = Dense(10, activation='softmax')(ext_model)

    # Create and compile the new extended model
    model = Model(inputs = vgg16_model.input, outputs = output)
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = optimizers.SGD(lr = 0.001, momentum = 0.9),
        metrics=['accuracy'])

    return model
