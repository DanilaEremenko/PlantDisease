import argparse
from pd_lib.conv_network import get_CNN, get_VGG16
from pd_lib.addition import save_model_to_json, get_full_model
import pd_lib.gui_reporter as gr
from pd_lib.img_proc import get_full_rect_image_from_pieces, draw_rect_on_array
from pd_lib.ui_cmd import get_input_int, get_stdin_answer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
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

    return model, json_list, new_model_type


def main():
    MAX_DRAW_IMG_SIZE = 1600
    #####################################################################
    # ----------------------- set data params ---------------------------
    #####################################################################
    ex_shape = (32, 32, 3)
    class_num = 2
    model, json_list, new_model_type = parse_args_for_train()

    #####################################################################
    # ----------------------- set train params --------------------------
    #####################################################################
    epochs_sum = 0
    lr = 1.0e-5

    verbose = True
    history_show = True
    title = 'train on ground'

    if model is None:
        if new_model_type == 'VGG16':
            model = get_VGG16(ex_shape, class_num)
            print("new VGG model created")
        else:
            model = get_CNN(ex_shape, class_num)
            print("new CNN model created")

    plot_model(model, show_shapes=True, to_file='model.png')

    checkpoint = ModelCheckpoint("model_ground.h5",
                                 monitor='val_loss',
                                 verbose=verbose,
                                 save_best_only=True,
                                 mode='auto')

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr), metrics=['accuracy'])

    class_1_num, class_2_num, ex_shape, x_train, y_train = \
        get_data_from_json_list(json_list, ex_shape, class_num)

    batch_size = int(y_train.shape[0] * 0.010)
    validation_split = 0.1
    full_history = {"acc": np.empty(0), "loss": np.empty(0)}

    print("batch_size = %.4f\nvalidation_split = %.4f\ntrain_size = %d\n" %
          (batch_size, validation_split, x_train.shape[0]))

    #####################################################################
    # ----------------------- creating train datagen --------------------
    #####################################################################
    train_generator = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split) \
        .flow(
        x=x_train,
        y=y_train,
        batch_size=batch_size
    )
    #####################################################################
    # ----------------------- train_model -------------------------------
    #####################################################################
    continue_train = True
    while continue_train:

        print("class_1_num = %d, class_2_num = %d" % (class_1_num, class_2_num))
        epochs = get_input_int("How many epochs?", 1, 100)

        history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=x_train.shape[0] / batch_size,
            epochs=epochs,
            shuffle=True,
            verbose=verbose,
            callbacks=[checkpoint, ]
        )

        full_history['acc'] = np.append(full_history['acc'], history.history['acc'])
        full_history['loss'] = np.append(full_history['loss'], history.history['loss'])

        gr.plot_history_separate_from_dict(history_dict=full_history,
                                           save_path_acc=None,
                                           save_path_loss=None,
                                           show=history_show,
                                           save=False
                                           )
        print("\naccuracy on train data\t %.f%%\n" % (history.history['acc'][epochs - 1]))

        i = 0
        x_draw = x_train.copy()
        class_1_ans = class_2_ans = 0
        right_ans = 0
        for y, mod_ans in zip(y_train[0:MAX_DRAW_IMG_SIZE], model.predict(x_train[0:MAX_DRAW_IMG_SIZE])):
            if y.__eq__([1, 0]).all():
                draw_rect_on_array(img_arr=x_draw[i], points=(1, 1, 31, 31), color=255)
            if mod_ans[1] > mod_ans[0]:
                draw_rect_on_array(img_arr=x_draw[i], points=(10, 10, 20, 20), color=255 * mod_ans[1])
                class_2_ans += 1
                if y.__eq__([1, 0]).all():
                    right_ans += 1

            elif mod_ans[0] > mod_ans[1]:
                draw_rect_on_array(img_arr=x_draw[i], points=(10, 10, 20, 20), color=0)
                class_1_ans += 1
                if y.__eq__([0, 1]).all():
                    right_ans += 1
            else:
                print("Too rare case")

            i += 1
        print("class_1_ans = %d, class_2_ans = %d\nright = %d (%.4f)" %
              (class_1_ans, class_2_ans, right_ans, right_ans / x_train.shape[0]))

        if get_stdin_answer("Show image of prediction?"):
            result_img = get_full_rect_image_from_pieces(x_draw[0:MAX_DRAW_IMG_SIZE])
            result_img.thumbnail(size=(1024, 1024))
            result_img.show()

        epochs_sum += epochs
        print("epochs: %d - %d" % (epochs_sum - epochs, epochs_sum))

        continue_train = get_stdin_answer(text="Continue?")

    save_model = get_stdin_answer(text='Save model?')

    if save_model:
        save_model_to_json(model, "models/model_ground_%d.json" % epochs_sum)
        model.save_weights('models/model_ground_%d.h5' % epochs_sum)


if __name__ == '__main__':
    main()
