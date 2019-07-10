from keras.models import model_from_json
from data_maker import get_data
from img_proc import plot_image_from_arr
from conv_network import get_CNN
import gui_reporter as gr


def get_model_from_json(path):
    json_string = ""
    for line in open(path, 'r').readlines(): json_string += line

    return model_from_json(json_string)


def train_on_dir(model, data, title, img_shape, max_ing_num=None, verbose=False, history_show=True):
    if not data.keys().__contains__("epochs"):
        raise Exception("no key epochs in train data")

    (x_train, y_train) = get_data(data, img_shape)

    # plot example
    plot_image_from_arr(x_train[0].transpose()[0].transpose())

    ##############################################################################
    # --------------------- model compiling & fitting ----------------------------
    ##############################################################################

    batch_size = int(y_train.shape[0] * 0.005)
    validation_split = 0.1

    print("############## training model on %s #######################" % title)

    history = model.fit \
            (
            x=x_train, y=y_train,
            epochs=data.get("epochs"),
            batch_size=batch_size, shuffle=True,
            validation_split=validation_split,
            verbose=verbose,
        )

    gr.plot_history_separte(history=history,
                            save_path_acc=None,
                            save_path_loss=None,
                            show=history_show,
                            save=False
                            )

    return model


def examine_on_dir(model, data, title, img_shape, max_img_num=None):
    (x_test, y_test) = get_data(data, img_shape, max_img_num=max_img_num)

    plot_image_from_arr(x_test[0].transpose()[0].transpose())

    verbose = False
    scores = model.evaluate(x_test, y_test, verbose=verbose)
    print("accuracy on %s = %.2f" % (title, scores[1]))
