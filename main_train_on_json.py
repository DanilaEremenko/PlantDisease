import argparse
from pd_lib.conv_network import get_CNN, get_VGG16
from pd_lib.addition import save_model_to_json, get_full_model
import pd_lib.gui_reporter as gr
from pd_lib.img_proc import get_full_rect_image_from_pieces, draw_rect_on_array
from pd_lib.ui_cmd import get_input_int, get_stdin_answer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import numpy as np
from pd_lib.data_maker import get_data_from_json_list


def parse_args_for_train():
    parser = argparse.ArgumentParser(description="Some description")

    parser.add_argument("-j", "--json_list", type=str, action="append",
                        help="json with train data")

    parser.add_argument("-w", "--weights_path", type=str, help="file with weigths of NN")

    parser.add_argument("-s", "--structure_path", type=str, help="file with structure of NN")

    parser.add_argument("-t", "--new_model_type", type=str, help="type of new model")

    args = parser.parse_args()

    weights_path = args.weights_path

    structure_path = args.structure_path

    model = get_full_model(json_path=structure_path, h5_path=weights_path, verbose=True)

    new_model_type = args.new_model_type

    json_list = args.json_list

    if json_list is None:
        raise Exception("Nor one json file passed")

    if new_model_type is None:
        new_model_type = 'cnn'

    return model, json_list, new_model_type


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_train_and_test(x_data, y_data, classes, validation_split):
    test_size = int(len(x_data) * validation_split)
    if test_size % 2 != 0:
        test_size += 1
    train_size = len(x_data) - test_size

    x_train = np.zeros(shape=(train_size, x_data.shape[1], x_data.shape[2], x_data.shape[3]))
    y_train = np.zeros(shape=(train_size, y_data.shape[1]))
    x_test = np.zeros(shape=(test_size, x_data.shape[1], x_data.shape[2], x_data.shape[3]))
    y_test = np.zeros(shape=(test_size, y_data.shape[1]))

    test_class_1_num = 0
    test_class_2_num = 0
    train_class_1_num = 0
    train_class_2_num = 0

    test_i = 0
    train_i = 0

    for i, (x, y) in enumerate(zip(x_data, y_data)):
        if (y == classes[0]).all() and test_class_1_num < test_size / 2:
            x_test[test_i] = x
            y_test[test_i] = y
            test_i += 1
            test_class_1_num += 1

        elif (y == classes[1]).all() and test_class_2_num < test_size / 2:
            x_test[test_i] = x
            y_test[test_i] = y
            test_i += 1
            test_class_2_num += 1

        elif (y == classes[0]).all():
            x_train[train_i] = x
            y_train[train_i] = y
            train_i += 1
            train_class_1_num += 1

        elif (y == classes[1]).all():
            x_train[train_i] = x
            y_train[train_i] = y
            train_i += 1
            train_class_2_num += 1

    return (x_train, y_train, train_class_1_num, train_class_2_num), \
           (x_test, y_test, test_class_1_num, test_class_2_num)


def main():
    MAX_DRAW_IMG_SIZE = 1600
    #####################################################################
    # ----------------------- set data params ---------------------------
    #####################################################################
    ex_shape = (32, 32, 3)
    class_num = 2
    model, json_list, new_model_type = parse_args_for_train()

    #####################################################################
    # ----------------------- model initializing ------------------------
    #####################################################################
    if model is None:
        if new_model_type.lower() == 'vgg16':
            model = get_VGG16(ex_shape, class_num)
            print("new VGG model created\n")
        else:
            model = get_CNN(ex_shape, class_num)
            print("new CNN model created\n")

    plot_model(model, show_shapes=True, to_file='model.png')

    #####################################################################
    # ----------------------- data initializing --------------------------
    #####################################################################
    validation_split = 0.1
    class_1_num, class_2_num, ex_shape, x_data, y_data = \
        get_data_from_json_list(json_list, ex_shape, class_num)

    # TODO need fix test and train data formation(it works incorrect and equality of class_examples_num isn't guaranteed
    (x_train, y_train, train_class_1_num, train_class_2_num), \
    (x_test, y_test, test_class_1_num, test_class_2_num) = get_train_and_test(x_data=x_data,
                                                                              y_data=y_data,
                                                                              classes=np.array([[1, 0], [0, 1]]),
                                                                              validation_split=validation_split)

    print("train_size = %d (class_1_num = %d, class_2_num = %d)" % (len(x_train), train_class_1_num, train_class_2_num))
    print("test_size = %d (class_1_num = %d, class_2_num = %d)\n" % (len(x_test), test_class_1_num, test_class_2_num))

    #####################################################################
    # ----------------------- set train params --------------------------
    #####################################################################
    epochs_sum = 0
    lr = 1.0e-5

    verbose = True
    history_show = True
    title = 'train on ground'

    batch_size = max(1, int(y_train.shape[0] * 0.010))
    validation_steps = max(1, int(y_test.shape[0] * 0.010))
    full_history = {"acc": np.empty(0), "loss": np.empty(0)}

    print("batch_size = %d\nvalidation_steps = %d\n" %
          (batch_size, validation_steps))

    #####################################################################
    # ----------------------- creating callbacks ------------------------
    #####################################################################
    checkpoint = ModelCheckpoint("model_ground.h5",
                                 monitor='val_loss',
                                 verbose=verbose,
                                 save_best_only=True,
                                 mode='auto')
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=1, patience=10,
                                   baseline=80, verbose=True)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])

    #####################################################################
    # ----------------------- creating train datagen --------------------
    #####################################################################
    train_generator = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    ) \
        .flow(
        x=x_train,
        y=y_train,
        batch_size=batch_size
    )
    validation_generator = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    ) \
        .flow(
        x=x_test,
        y=y_test,
        batch_size=batch_size
    )
    #####################################################################
    # ----------------------- train_model -------------------------------
    #####################################################################
    continue_train = True
    while continue_train:

        epochs = get_input_int("How many epochs?", 1, 100)

        history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=x_train.shape[0] / batch_size,
            validation_steps=validation_steps,
            validation_data=validation_generator,
            epochs=epochs,
            shuffle=True,
            verbose=verbose,
            callbacks=[checkpoint, early_stopping]
        )

        full_history['acc'] = np.append(full_history['acc'], history.history['acc'])
        full_history['loss'] = np.append(full_history['loss'], history.history['loss'])

        gr.plot_history_separate_from_dict(history_dict=full_history,
                                           save_path_acc=None,
                                           save_path_loss=None,
                                           show=history_show,
                                           save=False
                                           )
        print("\naccuracy on test data\t %.f%%\n" % (history.history['acc'][-1]))

        i = 0
        x_draw = x_train.copy()
        class_1_ans = class_2_ans = 0
        right_ans = 0
        for y, mod_ans in zip(y_train[0:MAX_DRAW_IMG_SIZE], model.predict(x_train[0:MAX_DRAW_IMG_SIZE])):
            if y.__eq__([1, 0]).all():
                draw_rect_on_array(img_arr=x_draw[i], points=(1, 1, 31, 31), color=255)
            if mod_ans[0] > mod_ans[1]:
                draw_rect_on_array(img_arr=x_draw[i], points=(10, 10, 20, 20), color=255)
                class_2_ans += 1
                if y.__eq__([1, 0]).all():
                    right_ans += 1

            elif mod_ans[1] > mod_ans[0]:
                draw_rect_on_array(img_arr=x_draw[i], points=(10, 10, 20, 20), color=0)
                class_1_ans += 1
                if y.__eq__([0, 1]).all():
                    right_ans += 1
            else:
                print("Too rare case")

            i += 1
        print("class_1_ans = %d, class_2_ans = %d\nright = %d (%.4f)" %
              (class_1_ans, class_2_ans, right_ans, right_ans / x_train.shape[0]))

        epochs_sum += len(history.history['acc'])
        print("epochs: %d - %d" % (epochs_sum - epochs, epochs_sum))

        if get_stdin_answer("Show image of prediction?"):
            result_img = get_full_rect_image_from_pieces(x_draw[0:MAX_DRAW_IMG_SIZE])
            result_img.thumbnail(size=(1024, 1024))
            result_img.show()

        if get_stdin_answer(text='Save model?'):
            save_model_to_json(model, "models/model_ground_%d.json" % epochs_sum)
            model.save_weights('models/model_ground_%d.h5' % epochs_sum)

        continue_train = get_stdin_answer(text="Continue?")


if __name__ == '__main__':
    main()
