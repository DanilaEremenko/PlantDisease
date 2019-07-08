import numpy as np
from keras.models import model_from_json
from data_maker import get_data
from img_proc import plot_image_from_arr
from keras.optimizers import Adam

if __name__ == '__main__':
    ##############################################################################
    # --------------------- model loading ----------------------------------------
    ##############################################################################
    json_string = ""
    for line in open("model.json", 'r').readlines(): json_string += line

    model = model_from_json(json_string)

    lr = 0.05
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

    ##############################################################################
    # --------------------- load testing data ------------------------------------
    ##############################################################################
    paths_to_dirs = (
        "PlantVillage/Tomato_Early_blight",
        "PlantVillage/Tomato_healthy",
        "PlantVillage/Tomato_Late_blight"
    )

    class_marks = np.array([
        (1, 0),
        (0, 1),
        (1, 0)
    ])

    # img_shape = get_imgs_shapes(paths_to_dirs[0]) # if parse without compressing
    img_shape = (32, 32, 3)  # if parse with compressing

    max_img_num = None

    (x_test, y_test) = get_data(paths_to_dirs, class_marks, img_shape, max_img_num=max_img_num)

    # plot example
    plot_image_from_arr(x_test[0].transpose()[0].transpose())

    ##############################################################################
    # --------------------- predicting & analyzing--------------------------------
    ##############################################################################
    verbose = True
    scores = model.evaluate(x_test, y_test, verbose=verbose)
    print("accuracy = %.2f" % scores[1])
