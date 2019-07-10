import numpy as np
from keras.optimizers import Adam
from addition import get_model_from_json, examine_on_dir, train_on_dir

if __name__ == '__main__':
    ##############################################################################
    # --------------------- model loading ----------------------------------------
    ##############################################################################
    model = get_model_from_json("model.json")
    lr = 0.05
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

    ##############################################################################
    # --------------------- choose testing data ----------------------------------
    ##############################################################################
    test_data = {
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
            ])
        },
        "tomato": {
            "data_dirs": (
                "PlantVillage/Tomato_Early_blight",
                "PlantVillage/Tomato_healthy",
                "PlantVillage/Tomato_Late_blight"
            ),
            "class_marks": np.array([
                (1, 0),
                (0, 1),
                (1, 0)
            ]),
            "epochs": 5
        },
        "pepper": {
            "data_dirs": (
                "PlantVillage/Pepper__bell___Bacterial_spot",
                "PlantVillage/Pepper__bell___healthy",
            ),
            "class_marks": np.array([
                (1, 0),
                (0, 1),
            ]),
            "epochs": 5
        }
    }

    train_data = {
        "tomato": test_data.get("tomato"),
        "pepper": test_data.get("pepper")
    }

    # img_shape = get_imgs_shapes(paths_to_dirs[0]) # if parse without compressing
    img_shape = (32, 32, 3)  # if parse with compressing

    ##############################################################################
    # ----------------------------- exam -----------------------------------------
    ##############################################################################
    for key in test_data.keys():
        examine_on_dir(
            model=model,
            data=test_data.get(key),
            title=key,
            img_shape=img_shape
        )

    ##############################################################################
    # ----------------------------- train ----------------------------------------
    ##############################################################################
    for key in train_data.keys():
        model = train_on_dir(model=model,
                             data=train_data.get(key),
                             title=key,
                             img_shape=img_shape,
                             verbose=False
                             )

    ##############################################################################
    # ----------------------------- exam after second learning -------------------
    ##############################################################################
    for key in test_data.keys():
        examine_on_dir(
            model=model,
            data=test_data.get(key),
            title=key,
            img_shape=img_shape
        )
