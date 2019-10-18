import json
import numpy as np

from pd_lib.img_proc import get_full_rect_image_from_pieces, draw_rect_on_array
from pd_lib.data_maker import multiple_class_examples, json_create
from pd_lib.ui_cmd import get_stdin_answer

if __name__ == '__main__':
    with open("Datasets/PotatoFields/plan_train/DJI_0246.json") as train_json_fp:
        train_json = dict(json.load(train_json_fp))
        class_1_num, class_2_num, x_train, y_train, img_shape = \
            train_json.get("class_1_num"), train_json.get("class_2_num"), \
            np.array(train_json.get("x_data"), dtype='uint8'), \
            np.array(train_json.get("y_data")), \
            train_json.get("img_shape")

    x_train, y_train = multiple_class_examples(x_train=x_train, y_train=y_train, class_for_multiple=[1, 0],
                                               use_noise=False, intensity_noise_list=(50, 150),
                                               use_deform=True, k_deform_list=(0.09, 0.10, 0.11, 0.12, 0.13),
                                               max_class_num=max([class_1_num, class_2_num]) * 2)

    class_2_num = y_train.shape[0] - class_1_num

    i = 0
    x_train_drawable = x_train.copy()
    for y in y_train:
        if ((y.__eq__([1, 0])).all()):
            draw_rect_on_array(x_train_drawable[i], (1, 1, 31, 31), 255)
        i += 1

    train_img_from_pieces = get_full_rect_image_from_pieces(x_train_drawable)
    train_img_from_pieces.show()

    out_json_path = "Datasets/PotatoFields/plan_train/DJI_0246_multiple.json"
    print("class_1_num = %d, class_2_num = %d" % (class_1_num, class_2_num))
    if get_stdin_answer("Save result to %s?" % out_json_path):
        json_create(path=out_json_path,
                    x_data=x_train, y_data=y_train,
                    img_shape=None,
                    class_1_num=class_1_num, class_2_num=class_2_num)
