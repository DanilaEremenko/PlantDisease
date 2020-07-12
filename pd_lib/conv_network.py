"""
Contains functions that return different CNN models
"""

from __future__ import print_function

from keras.applications import InceptionResNetV2, NASNetMobile, Xception, DenseNet121, MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, Convolution2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
import logging

logging.getLogger('tensorflow').disabled = True

MODELS_NAMES = ('DenseNet121', 'VGG16', 'MobileNetV2', 'InceptionResNetV2', 'Xception')
MODELS_KERAS = (DenseNet121, VGG16, MobileNetV2, InceptionResNetV2, Xception)


def get_model_by_name(name, input_shape, output_shape):
    for model_name, model_constructor in zip(
            MODELS_NAMES,
            MODELS_KERAS
    ):
        if name.lower() == model_name.lower():
            model = get_pre_trained(model_constructor, input_shape, output_shape)
            return model, model_name

    raise Exception('Undefined pretrained model name %s' % name)


def get_pre_trained(constructor, input_shape, output_shape):
    model = Sequential()

    model.add(constructor(include_top=False, weights='imagenet', input_shape=input_shape))

    model.add(Flatten())
    model.add(Dense(output_shape, activation='softmax'))

    return model


def get_CNN(input_shape, output_shape):
    model = Sequential()

    # 1 conv layer
    model.add(
        Conv2D(input_shape[0], kernel_size=3, padding='same',
               input_shape=input_shape,
               activation='relu'))

    # 2 conv layer
    model.add(Conv2D(input_shape[0], kernel_size=3, activation='relu'))

    # selection layer
    model.add(MaxPooling2D(data_format="channels_first",
                           pool_size=list(map(lambda z: max(2, int(z / 16)), input_shape[0:2]))))

    # regularization
    model.add(Dropout(0.25))

    # 3 conv layer
    model.add(Conv2D(input_shape[0] * 2, kernel_size=3, padding='same', activation='relu'))

    # 4 conv layer
    model.add(Conv2D(input_shape[0] * 2, kernel_size=3, activation='relu'))

    # selection layer
    model.add(MaxPooling2D(data_format="channels_first",
                           pool_size=list(map(lambda z: max(2, int(z / 32)), input_shape[0:2]))))

    # regularization
    model.add(Dropout(0.25))

    # 2D -> 1D
    model.add(Flatten())

    model.add(Dense(int(512), activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(output_shape, activation='softmax'))

    return model


def get_new_VGG16(input, out_neurons_num):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    return model
