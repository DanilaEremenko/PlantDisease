from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import logging

logging.getLogger('tensorflow').disabled = True


def get_CNN(img_shape, out_neurons_num):
    model = Sequential()

    # 1 conv layer
    model.add(
        Conv2D(img_shape[0], kernel_size=3, padding='same', input_shape=(img_shape[0], img_shape[1], img_shape[2]),
               activation='relu'))

    # 2 conv layer
    model.add(Conv2D(img_shape[0], kernel_size=3, activation='relu'))

    # selection layer
    model.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))

    # regularization
    model.add(Dropout(0.25))

    # 3 conv layer
    model.add(Conv2D(img_shape[0] * 2, kernel_size=3, padding='same', activation='relu'))

    # 4 conv layer
    model.add(Conv2D(img_shape[0] * 2, kernel_size=3, activation='relu'))

    # selection layer
    model.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))

    # regularization
    model.add(Dropout(0.25))

    # 2D -> 1D
    model.add(Flatten())

    model.add(Dense(int(img_shape[0] * img_shape[1] / 2), activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(out_neurons_num, activation='softmax'))

    return model
