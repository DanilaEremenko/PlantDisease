"""
Script for train new or early saved NN models via UI CMD
"""

import json

from pd_lib.conv_network import get_model_by_name
from pd_lib.keras_addition_ import save_model_to_json, get_full_model
from pd_lib.ui_cmd import get_input_int, get_stdin_answer

import pd_lib.data_maker as dmk

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import sys

from pd_main_part.preprocessors import get_preprocessor_by_name


def parse_args_for_train():
    # -------------------- initialize arguments ----------------------------------
    import argparse
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


def get_splited_subs(x_data, y_data, classes, validation_split):
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
        test_clasess[key]['max_num'] = int(classes[key]['num'] * validation_split)

    while sum(list(map(lambda x: x['max_num'], test_clasess.values()))) < test_size:
        test_clasess[list(test_clasess.keys())[0]]['max_num'] += 1

    test_i = 0
    train_i = 0

    for i, (x, y) in enumerate(zip(x_data, y_data)):
        for key in classes.keys():
            if (y == classes[key]['value']).all() \
                    and test_clasess[key]['num'] < test_clasess[key]['max_num'] \
                    and test_i < test_size:
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


def show_predict_on_window(x_data, y_data, y_predicted, classes):
    # TODO works incorrect
    from pd_gui.gui_fit_CNN import WindowShowPredictions
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)

    window_class_pctr = WindowShowPredictions(
        x_data=x_data,
        y_data=y_data,
        y_predicted=y_predicted,
        classes=classes
    )

    print("GUI EXITED WITH CODE = %d" % app.exec_())


def get_sub_arrays(x, y, classes, size=500):
    if x.shape[0] > size:
        (x1, y1, classes_1), (x2, y2, classes_2) = get_splited_subs(x, y, classes, validation_split=0.5)
        sub_1 = get_sub_arrays(x1, y1, classes_1)
        sub_2 = get_sub_arrays(x2, y2, classes_2)

        if isinstance(sub_1['x'], list):
            return {
                'x': [*sub_1['x'], *sub_2['x']],
                'y': [*sub_1['y'], *sub_2['y']],
                'classes': [*sub_1['classes'], *sub_2['classes']]
            }

        elif isinstance(sub_1['x'], np.ndarray):
            return {
                'x': [sub_1['x'], sub_2['x']],
                'y': [sub_1['y'], sub_2['y']],
                'classes': [classes_1, classes_2]
            }

        else:
            raise Exception("Whaaa")

    return {'x': x, 'y': y, 'classes': classes}


def get_flow_dict(flow_dir, json_path=None, data_dict=None):
    if json_path is not None:
        classes, img_shape, x_samples, y_samples = \
            dmk.json_big_load(json_path=json_path)
    elif data_dict is not None:
        classes = data_dict['classes']
        x_samples = data_dict['x']
        y_samples = data_dict['y']
    else:
        raise Exception('Data was not passed')

    from PIL import Image
    import os
    for key in classes.keys():
        classes[key]['value_num'] = dmk.get_num_from_pos(classes[key]['value'])
        cur_flow_dir = "%s/%d_%s" % (flow_dir, classes[key]['value_num'] + 1, key)
        if not os.path.isdir(cur_flow_dir):
            os.mkdir(cur_flow_dir)
            classes[key]['saved_num'] = 0
        else:
            print("%s already exist, exiting..." % cur_flow_dir)
            return {'classes': classes, 'flow_dir': flow_dir}

    for x, y in zip(x_samples, y_samples):
        for key in classes.keys():
            if (classes[key]['value'] == y).all():
                cur_flow_dir = "%s/%d_%s" % (flow_dir, classes[key]['value_num'] + 1, key)
                file_path = "%s/%d.JPG" % (cur_flow_dir, classes[key]['saved_num'] + 1)
                Image.fromarray(x).save(file_path)
                print("%s saved" % file_path)
                classes[key]['saved_num'] += 1

    return {'classes': classes, 'flow_dir': flow_dir}


################################################################################
# --------------------------------- MAIN ---------------------------------------
################################################################################
def main():
    #####################################################################
    # ----------------------- set data params ---------------------------
    #####################################################################
    if len(sys.argv) < 2:
        raise Exception('Path to config_train should be passed')
    with open(sys.argv[1]) as config_fp:
        config_dict = json.load(config_fp)

    with open(config_dict['data']['train_json']) as train_json_fp:
        train = json.load(train_json_fp)

    data_shape = (256, 256, 3)

    #####################################################################
    # ----------------------- preprocessor loading ----------------------
    #####################################################################
    if config_dict['preprocessor']['use']:
        preprocess_function = get_preprocessor_by_name(
            config_dict['preprocessor']['name'],
            config_dict['preprocessor']['args']).preprocess
    else:
        preprocess_function = None

    #####################################################################
    # ----------------------- data initializing --------------------------
    #####################################################################
    validation_split = 0.3
    evaluate_split = 0.3

    import copy

    test = copy.deepcopy(train)
    eval = copy.deepcopy(train)

    train['batch_size'] = 16
    test['batch_size'] = 8
    eval['batch_size'] = 16

    if 'dataframe' in train.keys():
        import pandas as pd
        train['df'] = pd.DataFrame(train['dataframe'])
        test['df'] = pd.DataFrame({'id': [], 'label': []})
        eval['df'] = pd.DataFrame({'id': [], 'label': []})

        for key in train['classes']:
            test['classes'][key]['num'] = int(train['classes'][key]['num'] * validation_split)
            eval['classes'][key]['num'] = int(test['classes'][key]['num'] * evaluate_split)

        def split_df(src_df, split_part, classes):
            res_df = pd.DataFrame({'id': [], 'label': []})
            for key in train['classes'].keys():
                res_df = res_df.append(src_df[src_df['label'] == key] \
                                           [:int(split_part * classes[key]['num'])])

            src_df = pd.merge(src_df, res_df, on=['id', 'label'], how='outer', indicator=True) \
                .query("_merge != 'both'") \
                .drop('_merge', axis=1) \
                .reset_index(drop=True)
            return src_df, res_df

        train['df'], test['df'] = split_df(train['df'], validation_split, train['classes'])
        test['df'], eval['df'] = split_df(test['df'], evaluate_split, test['classes'])

        del train['dataframe']
        del test['dataframe']
        del eval['dataframe']

        train['df'] = train['df'].replace(['фитофтороз', 'здоровый куст'], ['афитофтороз', 'яздоровый куст'])
        test['df'] = test['df'].replace(['фитофтороз', 'здоровый куст'], ['афитофтороз', 'яздоровый куст'])
        eval['df'] = eval['df'].replace(['фитофтороз', 'здоровый куст'], ['афитофтороз', 'яздоровый куст'])

        #####################################################################
        # ----------------------- creating generators from df ---------------
        #####################################################################
        train_generator = ImageDataGenerator(
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=preprocess_function
        ) \
            .flow_from_dataframe(
            dataframe=train['df'],
            x_col='id',
            y_col='label',
            target_size=(256, 256),
            color_mode='rgb',
            batch_size=train['batch_size']
        )

        validation_generator = ImageDataGenerator(
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=preprocess_function
        ) \
            .flow_from_dataframe(
            dataframe=test['df'],
            x_col='id',
            y_col='label',
            target_size=(256, 256),
            color_mode='rgb',
            batch_size=test['batch_size']
        )

        evaluate_generator = ImageDataGenerator(
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=preprocess_function
        ) \
            .flow_from_dataframe(
            dataframe=eval['df'],
            x_col='id',
            y_col='label',
            target_size=(256, 256),
            color_mode='rgb',
            batch_size=eval['batch_size']
        )



    else:
        ############################################################################
        # ------------------------- from dir ----------------------------------------
        ############################################################################
        train['flow_dir'] = '%s/train' % config_dict['data']['flow_dir']
        test['flow_dir'] = '%s/val' % config_dict['data']['flow_dir']
        eval['flow_dir'] = '%s/eval' % config_dict['data']['flow_dir']

        if config_dict['data']['create_flow_dir']:
            print('creating flow dir...')
            train['classes'], img_shape, train['x'], train['y'] = \
                dmk.json_big_load(config_dict['data']['train_json'])

            (train['x'], train['y'], train['classes']), \
            (test['x'], test['y'], test['classes']) = get_splited_subs(x_data=train['x'],
                                                                       y_data=train['y'],
                                                                       classes=train['classes'],
                                                                       validation_split=validation_split)
            (test['x'], test['y'], test['classes']), \
            (eval['x'], eval['y'], eval['classes']) = get_splited_subs(x_data=test['x'],
                                                                       y_data=test['y'],
                                                                       classes=test['classes'],
                                                                       validation_split=evaluate_split)

            data_shape = train['x'].shape[1:]

            train = get_flow_dict(data_dict=train, flow_dir=train['flow_dir'])
            test = get_flow_dict(data_dict=test, flow_dir=test['flow_dir'])
            eval = get_flow_dict(data_dict=eval, flow_dir=eval['flow_dir'])

            #####################################################################
            # ----------------------- creating generators from dir --------------
            #####################################################################
            train_generator = ImageDataGenerator(
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                preprocessing_function=preprocess_function
            ) \
                .flow_from_directory(
                directory=train['flow_dir'],
                target_size=(256, 256),
                color_mode='rgb',
                batch_size=train['batch_size']

            )

            validation_generator = ImageDataGenerator(
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                preprocessing_function=preprocess_function
            ) \
                .flow_from_directory(
                directory=test['flow_dir'],
                target_size=(256, 256),
                color_mode='rgb',
                batch_size=test['batch_size']

            )
            evaluate_generator = ImageDataGenerator(
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                preprocessing_function=preprocess_function
            ) \
                .flow_from_directory(
                directory=eval['flow_dir'],
                target_size=(256, 256),
                color_mode='rgb',
                batch_size=eval['batch_size']

            )

    # ------------------------------- weights setting --------------------------------------------
    class_weights = {}
    for class_info in train['classes'].values():
        class_weights[class_info['weight'][0]] = class_info['weight'][1]

    print("train['classes']  = %s" % str(train['classes']))
    print("test ['classes']  = %s" % str(test['classes']))
    print("eval ['classes']  = %s" % str(eval['classes']))

    print('class_weights = %s' % class_weights)

    #####################################################################
    # ----------------------- model initializing ------------------------
    #####################################################################
    if not config_dict['model']['create_new']:
        model = get_full_model(
            json_path=config_dict['model']['exist']['structure'],
            h5_path=config_dict['model']['exist']['weights'],
            verbose=True
        )
        model_name = 'NN'
    else:
        model, model_name = get_model_by_name(
            name=config_dict['model']['new']['type'],
            input_shape=data_shape,
            output_shape=len(train['classes'].keys())
        )
        print("new %s model created\n" % model_name)

    # plot_model(model, show_shapes=True, to_file='model.png')

    #####################################################################
    # ----------------------- set train params --------------------------
    #####################################################################
    epochs_sum = 0
    lr = 1.0e-5

    verbose = True
    history_show = True

    #####################################################################
    # ------------------ full history create/load -----------------------
    #####################################################################
    if not config_dict['model']['create_new'] and config_dict['model']['exist']['history']:
        with open(config_dict['model']['exist']['history']) as history_fp:
            full_history = json.load(history_fp)
        for key in full_history.keys():
            full_history[key] = np.array(full_history[key])
        print('history loaded from file')
    else:
        full_history = {"val_loss": np.empty(0), "val_accuracy": np.empty(0), "loss": np.empty(0),
                        "accuracy": np.empty(0)}
        print('new history initialized')

    print("train.batch_size = %d\ntest.batch_size = %d\neval.batch_size = %d\n" %
          (train['batch_size'], test['batch_size'], eval['batch_size']))

    #####################################################################
    # ----------------------- creating callbacks ------------------------
    #####################################################################
    baseline_dict = config_dict['fit']['baseline']
    print('BASELINES:%s' % str(baseline_dict))
    callbacks = [
        ModelCheckpoint("models/model_%s_best.h5" % model_name,
                        monitor='val_accuracy',
                        verbose=True,
                        save_best_only=True),
        # EarlyStopping(monitor='val_accuracy',
        #               patience=0,
        #               baseline=baseline_dict['test'],
        #               verbose=True,
        #               ),
    ]

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])

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
                steps_per_epoch=int(len(train['df']) / train['batch_size']),
                validation_data=validation_generator,
                validation_steps=int(len(test['df']) / test['batch_size']),
                epochs=epochs,
                shuffle=True,
                verbose=verbose,
                callbacks=callbacks,
                class_weight=class_weights
            )

            full_history['val_loss'] = np.append(full_history['val_loss'], history.history['val_loss'])
            full_history['val_accuracy'] = np.append(full_history['val_accuracy'], history.history['val_accuracy'])
            full_history['loss'] = np.append(full_history['loss'], history.history['loss'])
            full_history['accuracy'] = np.append(full_history['accuracy'], history.history['accuracy'])
            epochs_sum = len(full_history['accuracy'])

            #####################################################################
            # ----------------------- evaluate model ----------------------------
            #####################################################################
            print("\nacc        %.2f%%\n" % (history.history['accuracy'][-1] * 100), end='')
            print("val_acc      %.2f%%\n" % (history.history['val_accuracy'][-1] * 100), end='')

            if history.history['accuracy'][-1] < baseline_dict['train'] and epochs > len(history.history['accuracy']):
                bad_early_stop = True
                print("EarlyStopping by val_acc without acc, continue...")
                continue
            bad_early_stop = False

            epochs = len(history.history['accuracy'])

            # gr.plot_train_test_from_history(history_dict=history.history, show=True)
        eval['loss'], eval['accuracy'] = model.evaluate_generator(
            generator=evaluate_generator,
            steps=int(len(eval['df']) / eval['batch_size'])
        )
        print("eval_acc     %.2f%%\n" % (eval['accuracy'] * 100))

        print("epochs: %d - %d" % (epochs_sum - epochs, epochs_sum))

        #####################################################################
        # ----------------------- CMD UI ------------------------------------
        #####################################################################
        if get_stdin_answer("Show image of prediction?"):
            x, y = evaluate_generator.next()
            show_predict_on_window(
                x_data=np.array(x, 'uint8'),
                y_data=y,
                y_predicted=model.predict(x),
                classes=eval['classes']
            )
        if get_stdin_answer(text='Save model?'):
            save_model_to_json(model, "models/model_%s_%d.json" % (model_name, epochs_sum))
            model.save_weights('models/model_%s_%d.h5' % (model_name, epochs_sum))
            with open('models/model_%s_%d_history.json' % (model_name, epochs_sum), "w") as fp:
                json.dump(
                    obj=dict(zip(('val_loss', 'val_accuracy', 'loss', 'accuracy'),
                                 list(map(lambda x: x.tolist(), full_history.values())))),
                    fp=fp
                )

        continue_train = get_stdin_answer(text="Continue?")


if __name__ == '__main__':
    main()
