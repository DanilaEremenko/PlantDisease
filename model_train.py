from conv_network import get_CNN
from data_maker import get_data
from img_proc import plot_image_from_arr
import gui_reporter as gr
import numpy as np

def save_to_json(model, path):
    json_string = model.to_json()
    json_file = open(path, "w")
    json_file.write(json_string)
    json_file.close()


if __name__ == '__main__':
    ##############################################################################
    # --------------------- train data making ------------------------------------
    ##############################################################################
    paths_to_dirs = (
        "PlantVillage/Potato___Early_blight",
        "PlantVillage/Potato___healthy",
        "PlantVillage/Potato___Late_blight"
    )

    class_marks = np.array([
        (1,0),
        (0,1),
        (1,0)
    ])

    # img_shape = get_imgs_shapes(paths_to_dirs[0]) # if parse without compressing
    img_shape = (32, 32, 3)  # if parse with compressing

    max_img_num = None

    (x_train, y_train) = get_data(paths_to_dirs, class_marks, img_shape, max_img_num=max_img_num)

    # plot example
    plot_image_from_arr(x_train[0].transpose()[0].transpose())

    ##############################################################################
    # --------------------- model compiling & fitting ----------------------------
    ##############################################################################

    verbose = True
    epochs = 20
    batch_size = 16
    validation_split = 0.1
    show = True
    save = not show
    lr = 0.05

    model = get_CNN(img_shape, out_neurons_num=class_marks[0].shape[0], lr=lr)

    history = model.fit \
            (
            x=x_train, y=y_train,
            epochs=epochs,
            batch_size=batch_size, shuffle=True,
            validation_split=validation_split,
            verbose=verbose,
        )

    gr.plot_history_separte(history=history,
                            save_path_acc=None,
                            save_path_loss=None,
                            show=show,
                            save=save
                            )

    ##############################################################################
    # --------------------- model saving -----------------------------------------
    ##############################################################################
    save_to_json(model, "model.json")

    model.save_weights('model_weights.h5')
