# coding=utf-8
from keras.models import model_from_json
from pd_lib import img_proc as img_pr
from pd_lib import gui_reporter as gr
from pd_lib import data_maker as dmk


################################################################################
# --------------------------------- saving & loading models --------------------
################################################################################
def save_model_to_json(model, path):
    json_file = open(path, "w")
    json_file.write(model.to_json())
    json_file.close()


def get_model_from_json(path):
    json_string = ""
    for line in open(path, 'r').readlines(): json_string += line

    return model_from_json(json_string)


def get_full_model(json_path, h5_path, verbose=False):
    if json_path is None or h5_path is None:
        return None

    model = get_model_from_json(json_path)
    model.load_weights(h5_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if verbose:
        print("model loaded successfully")
    return model


################################################################################
# --------------------------------- training and testing ----------------
################################################################################
def train_on_json(model, json_list, epochs, img_shape, class_num, verbose=False, history_show=True):
    class_1_num, class_2_num, img_shape, x_train, y_train = dmk.get_data_from_json_list(json_list, img_shape, class_num)

    batch_size = int(y_train.shape[0] * 0.005)
    validation_split = 0.1

    history = model.fit(
        x=x_train, y=y_train,
        epochs=epochs,
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


def predict_and_localize_on_image(model, cropped_data, image, color_1, color_2, verbose=False):
    for curr_window, coord in zip(cropped_data["x_data"], cropped_data["x_coord"]):
        curr_window.shape = (1, curr_window.shape[0], curr_window.shape[1], curr_window.shape[2])
        pred = model.predict(curr_window)
        if verbose:
            print("%d %d" % (pred[0][0], pred[0][1]))
        if pred[0][0] > pred[0][1]:
            image = img_pr.draw_rect_on_image(image, (coord[0] + 1, coord[1] + 1, coord[2] - 1, coord[3] - 1), color_1)
        elif pred[0][0] < pred[0][1]:
            image = img_pr.draw_rect_on_image(image, (coord[0] + 1, coord[1] + 1, coord[2] - 1, coord[3] - 1), color_2)
        else:
            print("Too rare case")

    return image
