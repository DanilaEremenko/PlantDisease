import numpy as np
from pd_lib.addition import train_on_dir
from pd_lib.conv_network import get_CNN




if __name__ == '__main__':
    ##############################################################################
    # --------------------- train data making ------------------------------------
    ##############################################################################
    train_data = {
        "potato": {
            "data_dirs": (
                "Datasets/PlantVillage/Potato___Early_blight",
                "Datasets/PlantVillage/Potato___healthy",
                "Datasets/PlantVillage/Potato___Late_blight"
            ),
            "class_marks": np.array([
                (1, 0),
                (0, 1),
                (1, 0)
            ]),
            "epochs": 20
        }
    }

    # img_shape = get_imgs_shapes(paths_to_dirs[0]) # if parse without compressing
    img_shape = (32, 32, 3)  # if parse with compressing
    lr = 0.05
    out_neurons_num = train_data.get("potato").get("class_marks")[0].shape[0]

    model = get_CNN(img_shape, out_neurons_num=out_neurons_num, lr=lr)

    for key in train_data.keys():
        model = train_on_dir(model=model,
                             data=train_data.get(key),
                             title=key,
                             img_shape=img_shape,
                             verbose=True
                             )

    ##############################################################################
    # --------------------- model saving -----------------------------------------
    ##############################################################################
    save_to_json(model, "models/model_potato_%d.json" % train_data.get("potato").get("epochs"))

    model.save_weights('models/model_potato_%d.h5' % train_data.get("potato").get("epochs"))
