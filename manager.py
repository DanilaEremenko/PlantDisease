from conv_network import get_CNN
from data_maker import get_data
from keras.utils import np_utils

if __name__ == '__main__':
    ##############################################################################
    # --------------------- model building ---------------------------------------
    ##############################################################################
    #model = get_CNN()

    ##############################################################################
    # --------------------- train data making ------------------------------------
    ##############################################################################
    paths_to_dirs = (
        "PlantVillage/Potato___Early_blight",
        "PlantVillage/Potato___healthy",
        "PlantVillage/Potato___Late_blight"
    )

    class_marks = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1)
    )

    max_img_num = 10

    (x_train, y_train) = get_data(paths_to_dirs, class_marks, max_img_num=10)

    ##############################################################################
    # --------------------- model compiling & fitting ----------------------------
    ##############################################################################
    # verbose = True
    # epochs = 3
    # batch_size = 32
    # validation_split = 0.1
    #
    # history = model.fit \
    #         (
    #         x=x_train, y=y_train,
    #         epochs=epochs,
    #         batch_size=batch_size, shuffle=True,
    #         validation_split=validation_split,
    #         verbose=verbose
    #     )
    #
    # scores = model.evaluate(x_test, y_test, verbose=verbose)
    #
    # print("accuracy = %.2f" % scores[1])
