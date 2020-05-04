import sys

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import ReLU
from keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import numpy as np
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from keras_unet.models import custom_unet
from pd_lib.gui_reporter import plot_history_separate_from_dict
from pd_lib.keras_addition_ import save_model_to_json, get_full_model
from pd_lib.ui_cmd import get_stdin_answer, get_input_int


def show_predict_on_window(x_data, y_data, y_predicted):
    # TODO works incorrect
    from PyQt5 import QtWidgets
    from pd_gui.gui_fitting_unet import WindowShowUnetFitting

    app = QtWidgets.QApplication(sys.argv)

    window_class_pctr = WindowShowUnetFitting(x_data, y_data, y_predicted)

    print("GUI EXITED WITH CODE = %d" % app.exec_())


def main():
    train_x = np.empty(0, dtype='uint8')
    train_y = np.empty(0, dtype='uint8')
    batch_size = 4

    x_dir = 'DataForSegmentator/input'
    y_dir = 'DataForSegmentator/output_filters'

    samples_num = len(os.listdir(x_dir))

    for x_path, y_path in zip(sorted(os.listdir(x_dir)), sorted(os.listdir(y_dir))):
        train_x = np.append(train_x, np.array(Image.open("%s/%s" % (x_dir, x_path))))
        train_y = np.append(train_y, np.array(Image.open("%s/%s" % (y_dir, y_path))))

    train_x.shape = (samples_num, 256, 256, 3)
    train_y.shape = (samples_num, 256, 256, 3)

    if not os.path.isfile('train_unet_x.npy'):
        np.save('train_unet_x', train_x)
    train_x = np.memmap('train_unet_x.npy', shape=train_x.shape, offset=128)

    x_generator = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,

    ) \
        .flow(
        x=train_x,
        batch_size=batch_size,
        seed=42
    )
    y_generator = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,

    ) \
        .flow(
        x=train_y,
        batch_size=batch_size,
        seed=42
    )

    train_generator = zip(x_generator, y_generator)

    model = custom_unet(
        input_shape=train_x.shape[1:],
        use_batch_norm=False,
        num_classes=3,
        filters=32,
        dropout=0.2,
        output_activation='relu'
    )

    # model = get_full_model(json_path='models/model_unet_70.json', h5_path='models/model_unet_70.h5')

    model.compile(optimizer=Adam(lr=1e-8), loss='mae', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint("model_unet.h5",
                        monitor='acc',
                        verbose=True,
                        save_best_only=True),
        # EarlyStopping(monitor='acc',
        #               patience=0,
        #               baseline=90,
        #               verbose=True,
        #               ),
    ]

    full_history = {"acc": np.empty(0), "loss": np.empty(0)}

    continue_train = True
    epochs_sum = 0
    while continue_train:

        epochs = get_input_int("How many epochs?", 0, 100)

        history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=int(samples_num / batch_size),
            # validation_steps=train_x.shape[0] / batch_size,
            epochs=epochs,
            shuffle=True,
            verbose=True,
            callbacks=callbacks,
        )

        if epochs != 0:
            full_history['acc'] = np.append(full_history['acc'], history.history['accuracy'])
            full_history['loss'] = np.append(full_history['loss'], history.history['loss'])
            epochs_sum = len(full_history['acc'])

            #####################################################################
            # ----------------------- evaluate model ----------------------------
            #####################################################################
            print("\nacc        %.2f%%\n" % (history.history['accuracy'][-1] * 100), end='')

            epochs = len(history.history['accuracy'])

            plot_history_separate_from_dict(history_dict=full_history,
                                            save_path_acc=None,
                                            save_path_loss=None,
                                            show=True,
                                            save=False
                                            )

        print("epochs: %d - %d" % (epochs_sum - epochs, epochs_sum))

        #####################################################################
        # ----------------------- CMD UI ------------------------------------
        #####################################################################
        if get_stdin_answer("Show image of prediction?"):
            show_predict_on_window(train_x, train_y, np.array(model.predict(train_x), dtype='uint8'))

        if get_stdin_answer(text='Save model?'):
            save_model_to_json(model, "models/model_unet_%d.json" % (epochs_sum))
            model.save_weights('models/model_unet_%d.h5' % (epochs_sum))

        continue_train = get_stdin_answer(text="Continue?")


if __name__ == '__main__':
    main()
