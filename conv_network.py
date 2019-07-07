from __future__ import print_function

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam

import logging

logging.getLogger('tensorflow').disabled = True

import sys

sys.path.append("../NeuralNetwork/ADDITIONAL")

from keras.datasets import cifar10

np.random.seed(42)


def get_CNN():
    model = Sequential()

    # 1 conv layer
    model.add(Conv2D(32, kernel_size=3, padding='same', input_shape=(32, 32, 3), activation='relu'))

    # 2 conv layer
    model.add(Conv2D(32, kernel_size=3, activation='relu'))

    # selection layer
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    # regularization
    model.add(Dropout(0.25))

    # 3 conv layer
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))

    # 4 conv layer
    model.add(Conv2D(64, kernel_size=3, activation='relu'))

    # selection layer
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    # regularization
    model.add(Dropout(0.25))

    # 2D -> 1D
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])

    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
