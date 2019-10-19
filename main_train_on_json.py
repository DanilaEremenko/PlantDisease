from pd_lib.conv_network import get_CNN
from pd_lib.addition import save_to_json
import pd_lib.gui_reporter as gr
import pd_lib.data_maker as dmk
from pd_lib.img_proc import get_full_rect_image_from_pieces, draw_rect_on_array
from pd_lib.ui_cmd import get_input_int, get_stdin_answer
from keras.optimizers import Adam
import numpy as np

if __name__ == '__main__':
    #####################################################################
    # ----------------------- set data params ---------------------------
    #####################################################################
    ex_shape = (32, 32, 3)
    class_num = 2
    json_list = ["Datasets/PotatoFields/plan_train/DJI_0246_multiple.json"]

    #####################################################################
    # ----------------------- set train params --------------------------
    #####################################################################
    epochs = epochs_sum = 0
    lr = 0.15

    verbose = True
    history_show = True
    title = 'train on ground'

    model = get_CNN(ex_shape, class_num)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

    class_1_num, class_2_num, ex_shape, x_train, y_train = \
        dmk.get_data_from_json_list(json_list, ex_shape, class_num)

    batch_size = int(y_train.shape[0] * 0.010)
    validation_split = 0.1
    full_history = {"acc": np.empty(0), "loss": np.empty(0)}

    print("batch_size = %.4f\nvalidation_split = %.4f\ntrain_size = %d\n" %
          (batch_size, validation_split, x_train.shape[0]))
    #####################################################################
    # ----------------------- train_model -------------------------------
    #####################################################################
    continue_train = True
    while continue_train:

        epochs = get_input_int("How many epochs?", 1, 20)

        history = model.fit(
            x=x_train, y=y_train,
            epochs=epochs,
            batch_size=batch_size, shuffle=True,
            validation_split=validation_split,
            verbose=verbose,
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
        for y, mod_ans in zip(y_train, model.predict(x_train)):
            if y.__eq__([1, 0]).all():
                draw_rect_on_array(img_arr=x_draw[i], points=(1, 1, 31, 31), color=255)
            if mod_ans.__eq__([1, 0]).all():
                draw_rect_on_array(img_arr=x_draw[i], points=(10, 10, 20, 20), color=0)
            elif mod_ans.__eq__([0, 1]).all():
                draw_rect_on_array(img_arr=x_draw[i], points=(10, 10, 20, 20), color=255)

            i += 1

        if get_stdin_answer("Show image of prediction?"):
            result_img = get_full_rect_image_from_pieces(x_draw)
            result_img.thumbnail(size=(1024, 1024))
            result_img.show()

        epochs_sum += epochs
        print("epochs: %d - %d" % (epochs_sum - epochs, epochs_sum))

        continue_train = get_stdin_answer(text="Continue?")

    save_model = get_stdin_answer(text='Save model?')

    if save_model:
        save_to_json(model, "models/model_ground_%d.json" % epochs_sum)
        model.save_weights('models/model_ground_%d.h5' % epochs_sum)
