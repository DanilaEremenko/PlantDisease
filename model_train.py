import numpy as np
from addition import train_on_dir
from conv_network import get_CNN


def save_to_json(model, path):
    json_string = model.to_json()
    json_file = open(path, "w")
    json_file.write(json_string)
    json_file.close()


if __name__ == '__main__':
    ##############################################################################
    # --------------------- train data making ------------------------------------
    ##############################################################################
    train_data = {
        "potato": {
            "data_dirs": (
                "PlantVillage/Potato___Early_blight",
                "PlantVillage/Potato___healthy",
                "PlantVillage/Potato___Late_blight"
            ),
            "class_marks": np.array([
                (1, 0),
                (0, 1),
                (1, 0)
            ]),
            "epochs": 30
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
    save_to_json(model, "model.json")

    model.save_weights('model_weights.h5')
