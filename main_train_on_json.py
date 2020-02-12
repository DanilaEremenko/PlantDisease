import argparse
import json

from pd_lib.conv_network import get_CNN, get_VGG16
from pd_lib.addition import save_model_to_json, get_full_model
from pd_lib.img_proc import get_full_rect_image_from_pieces, draw_rect_on_array
from pd_lib.ui_cmd import get_input_int, get_stdin_answer

import pd_lib.data_maker as dmk
import pd_lib.gui_reporter as gr

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import numpy as np

import sys


def parse_args_for_train():
    # -------------------- initialize arguments ----------------------------------
    parser = argparse.ArgumentParser(description="Some description")

    parser.add_argument("-j", "--json_list", type=str, action="append",
                        help="path to json with train data")

    parser.add_argument("-e", "--evaluate_list", type=str, action="append",
                        help="path to json with evaluate data")

    parser.add_argument("-w", "--weights_path", type=str, help="file with weigths of NN")

    parser.add_argument("-s", "--structure_path", type=str, help="file with structure of NN")

    parser.add_argument("-t", "--new_model_type", type=str, help="type of new model")

    # -------------------- parsing arguments ----------------------------------
    args = parser.parse_args()

    json_list = args.json_list

    evaluate_list = args.evaluate_list

    weights_path = args.weights_path
    structure_path = args.structure_path
    model = get_full_model(json_path=structure_path, h5_path=weights_path, verbose=True)

    new_model_type = args.new_model_type
    # -------------------- validation check ----------------------------------
    if json_list is None:
        raise Exception("Nor one json file passed")

    if evaluate_list is None:
        raise Exception("Nor one evaluate json file passed")

    if new_model_type is None:
        new_model_type = 'CNN'

    return model, json_list, evaluate_list, new_model_type


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_train_and_test(x_data, y_data, classes, validation_split):
    test_size = int(len(x_data) * validation_split)
    while test_size % len(classes.keys()) != 0:
        test_size += 1
    train_size = len(x_data) - test_size

    x_train = np.zeros(shape=(train_size, x_data.shape[1], x_data.shape[2], x_data.shape[3]), dtype='uint8')
    y_train = np.zeros(shape=(train_size, y_data.shape[1]))
    x_test = np.zeros(shape=(test_size, x_data.shape[1], x_data.shape[2], x_data.shape[3]), dtype='uint8')
    y_test = np.zeros(shape=(test_size, y_data.shape[1]))

    train_clasess = {}
    test_clasess = {}
    for key in classes.keys():
        train_clasess[key] = classes[key].copy()
        train_clasess[key]['num'] = 0

        test_clasess[key] = classes[key].copy()
        test_clasess[key]['num'] = 0

    test_i = 0
    train_i = 0

    for i, (x, y) in enumerate(zip(x_data, y_data)):
        for key in classes.keys():
            if (y == classes[key]['value']).all() and test_clasess[key]['num'] < test_size / len(classes.keys()):
                x_test[test_i] = x
                y_test[test_i] = y
                test_i += 1
                test_clasess[key]['num'] += 1
                break

            elif (y == classes[key]['value']).all():
                x_train[train_i] = x
                y_train[train_i] = y
                train_i += 1
                train_clasess[key]['num'] += 1
    return (x_train, y_train, train_clasess), \
           (x_test, y_test, test_clasess)


def predict_and_draw_on_data(model, x, y):
    i = 0
    res = {'x_draw': x.copy(), 'class_1_ans': 0, 'class_2_ans': 0, 'right_ans': 0}

    for y, mod_ans in zip(y, model.predict(x)):
        if y.__eq__([1, 0]).all():
            draw_rect_on_array(img_arr=res['x_draw'][i], points=(1, 1, 31, 31), color=255)
        if mod_ans[0] > mod_ans[1]:
            draw_rect_on_array(img_arr=res['x_draw'][i], points=(10, 10, 20, 20), color=255)
            res['class_2_ans'] += 1
            if y.__eq__([1, 0]).all():
                res['right_ans'] += 1

        elif mod_ans[1] > mod_ans[0]:
            draw_rect_on_array(img_arr=res['x_draw'][i], points=(10, 10, 20, 20), color=0)
            res['class_1_ans'] += 1
            if y.__eq__([0, 1]).all():
                res['right_ans'] += 1
        else:
            print("Too rare case")

        i += 1

    return res


def show_predict_on_window(model, x_data, y_data, classes):
    # TODO works incorrect
    from pd_gui.gui_train_examine import WindowMultipleExamples
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)

    window_class_pctr = WindowMultipleExamples(
        model=model,
        x_data=x_data,
        y_data=y_data,
        classes=classes
    )

    print("GUI EXITED WITH CODE = %d" % app.exec_())


################################################################################
# --------------------------------- MAIN ---------------------------------------
################################################################################
def main():
    MAX_DRAW_IMG_SIZE = 1600
    #####################################################################
    # ----------------------- set data params ---------------------------
    #####################################################################
    ex_shape = (32, 32, 3)
    if len(sys.argv) < 2:
        raise Exception('Path to config_train should be passed')
    with open(sys.argv[1]) as config_fp:
        config_dict = json.load(config_fp)

    json_list = config_dict['data']['train_list']
    eval_list = config_dict['data']['eval_list']
    # model, json_list, evaluate_list, new_model_type = parse_args_for_train()

    #####################################################################
    # ----------------------- data initializing --------------------------
    #####################################################################
    validation_split = 0.1
    train = {}
    test = {}
    eval = {}

    train['classes'], train["x"], train["y"] = \
        dmk.get_data_from_json_list(json_list, ex_shape)

    eval['classes'], eval["x"], eval["y"] = \
        dmk.get_data_from_json_list(eval_list, ex_shape)

    eval['x'] = np.array(eval['x'], dtype='uint8')

    (train["x"], train["y"], train['classes']), \
    (test["x"], test["y"], test['classes']) = get_train_and_test(x_data=train["x"],
                                                                 y_data=train["y"],
                                                                 classes=train['classes'],
                                                                 validation_split=validation_split)

    print("train = %s" % str(train['classes']))
    print("test  = %s" % str(test['classes']))
    print("eval  = %s" % str(eval['classes']))

    #####################################################################
    # ----------------------- model initializing ------------------------
    #####################################################################
    if not config_dict['model']['create_new']:
        model = get_full_model(
            json_path=config_dict['model']['exist']['structure'],
            h5_path=config_dict['model']['exist']['weights'],
            verbose=True
        )
        new_model_type = 'NN'
    elif config_dict['model']['new']['type'] == 'vgg16':
        model = get_VGG16(ex_shape, len(train['classes'].keys()))
        print("new VGG16 model created\n")
        new_model_type = 'VGG16'
    else:
        model = get_CNN(ex_shape, len(train['classes'].keys()))
        print("new CNN model created\n")
        new_model_type = 'CNN'

    plot_model(model, show_shapes=True, to_file='model.png')

    #####################################################################
    # ----------------------- set train params --------------------------
    #####################################################################
    epochs_sum = 0
    lr = 1.0e-5

    verbose = True
    history_show = True

    train['batch_size'] = max(1, int(train["y"].shape[0] * 0.010))
    test['batch_size'] = max(1, int(test["y"].shape[0] * 0.010))
    eval['batch_size'] = max(1, int(eval["y"].shape[0] * 0.010))
    full_history = {"acc": np.empty(0), "loss": np.empty(0)}

    print("train.batch_size = %d\ntest.batch_size = %d\neval.batch_size = %d\n" %
          (train['batch_size'], test['batch_size'], eval['batch_size']))

    #####################################################################
    # ----------------------- creating callbacks ------------------------
    #####################################################################
    baseline_dict = config_dict['fit']['baseline']
    print('BASELINES:%s' % str(baseline_dict))
    callbacks = [
        ModelCheckpoint("model_ground.h5",
                        monitor='val_acc',
                        verbose=True,
                        save_best_only=True),
        EarlyStopping(monitor='val_acc',
                      patience=0,
                      baseline=baseline_dict['test'],
                      verbose=True,
                      ),
    ]

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
        x=train['x'],
        y=train['y'],
        batch_size=train['batch_size']
    )
    validation_generator = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    ) \
        .flow(
        x=test['x'],
        y=test['y'],
        batch_size=test['batch_size']
    )

    evaluate_generator = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    ) \
        .flow(
        x=eval['x'],
        y=eval['y'],
        batch_size=eval['batch_size']
    )
    #####################################################################
    # ----------------------- train_model -------------------------------
    #####################################################################
    continue_train = True
    bad_early_stop = False
    while continue_train:

        if not bad_early_stop:
            epochs = get_input_int("How many epochs?", 0, 100)

        if epochs != 0:
            history = model.fit_generator(
                generator=train_generator,
                steps_per_epoch=train['x'].shape[0] / train['batch_size'],
                validation_steps=test['batch_size'],
                validation_data=validation_generator,
                epochs=epochs,
                shuffle=True,
                verbose=verbose,
                callbacks=callbacks
            )

            full_history['acc'] = np.append(full_history['acc'], history.history['acc'])
            full_history['loss'] = np.append(full_history['loss'], history.history['loss'])
            epochs_sum = len(full_history['acc'])

            #####################################################################
            # ----------------------- evaluate model ----------------------------
            #####################################################################
            eval['loss'], eval['acc'] = model.evaluate_generator(
                generator=evaluate_generator,
                steps=train['x'].shape[0] / eval['batch_size']
            )

            print("\nacc        %.2f%%\n" % (history.history['acc'][-1] * 100), end='')
            print("val_acc      %.2f%%\n" % (history.history['val_acc'][-1] * 100), end='')
            print("eval_acc     %.2f%%\n" % (eval['acc'] * 100))

            if history.history['acc'][-1] < baseline_dict['train'] and epochs > len(history.history['acc']):
                bad_early_stop = True
                print("EarlyStopping by val_acc without acc, continue...")
                continue
            bad_early_stop = False

            epochs = len(history.history['acc'])

            gr.plot_history_separate_from_dict(history_dict=full_history,
                                               save_path_acc=None,
                                               save_path_loss=None,
                                               show=history_show,
                                               save=False
                                               )

        print("epochs: %d - %d" % (epochs_sum - epochs, epochs_sum))

        #####################################################################
        # ----------------------- CMD UI ------------------------------------
        #####################################################################
        if get_stdin_answer("Show image of prediction?"):
            show_predict_on_window(
                model=model,
                x_data=eval['x'],
                y_data=eval['y'],
                classes=eval['classes']
            )
        if get_stdin_answer(text='Save model?'):
            save_model_to_json(model, "models/model_ground_%s_%d.json" % (new_model_type, epochs_sum))
            model.save_weights('models/model_ground_%s_%d.h5' % (new_model_type, epochs_sum))

        continue_train = get_stdin_answer(text="Continue?")


if __name__ == '__main__':
    main()
