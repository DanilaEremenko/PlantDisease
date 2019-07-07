from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
import logging

logging.getLogger('tensorflow').disabled = True


def get_CNN(img_shape, class_num, lr=0.1):
    model = Sequential()

    # 1 conv layer
    model.add(
        Conv2D(img_shape[0], kernel_size=3, padding='same', input_shape=(img_shape[0], img_shape[1], img_shape[2]),
               activation='relu'))

    # 2 conv layer
    model.add(Conv2D(img_shape[0], kernel_size=3, activation='relu'))

    # selection layer
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    # regularization
    model.add(Dropout(0.25))

    # 3 conv layer
    model.add(Conv2D(img_shape[0] * 2, kernel_size=3, padding='same', activation='relu'))

    # 4 conv layer
    model.add(Conv2D(img_shape[0] * 2, kernel_size=3, activation='relu'))

    # selection layer
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    # regularization
    model.add(Dropout(0.25))

    # 2D -> 1D
    model.add(Flatten())

    model.add(Dense(img_shape[0] * img_shape[1] / 2, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(class_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

    return model
