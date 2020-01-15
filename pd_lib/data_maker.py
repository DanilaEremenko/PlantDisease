import numpy as np
import os
from PIL import Image
from pd_lib import img_proc as img_pr
from pd_geo import exif_parser as ep
import json


################################################################################
# --------------------------------- for predict on image -----------------------
################################################################################
def get_x_from_croped_img(path_img_in, img_shape, window_shape, step=1.0, color=255, path_out_dir=None):
    if path_out_dir != None and not os.path.isdir(path_out_dir):
        raise Exception("No such directory %s" % path_out_dir)

    full_img = Image.open(path_img_in)
    full_img.thumbnail(img_shape)
    img_exif = ep.get_exif_data(full_img)

    draw_image = full_img

    p1_x, p1_y, p2_x, p2_y = 0, 0, window_shape[0], window_shape[1]
    i = 0
    x_data = np.empty(0, dtype='uint8')
    x_coord = np.empty(0, dtype='uint8')
    longitudes = []  # TODO add caluclating
    latitudes = []  # TODO add caluclating

    while p2_y <= full_img.size[1]:
        while p2_x <= full_img.size[0]:
            if path_out_dir != None:
                img_pr.crop_multiply_data(img=full_img,
                                          name="%d" % i,
                                          crop_area=(p1_x, p1_y, p2_x, p2_y),
                                          path_out_dir=path_out_dir
                                          )

            curr_x = np.asarray(full_img.crop((p1_x, p1_y, p2_x, p2_y)))
            x_data = np.append(x_data, curr_x)
            x_coord = np.append(x_coord, (p1_x, p1_y, p2_x, p2_y))

            draw_image = img_pr.draw_rect_on_image(draw_image, (p1_x, p1_y, p2_x, p2_y), color=color)

            p1_x += int(window_shape[0] * step)
            p2_x += int(window_shape[0] * step)
            i += 1
        p1_x = 0
        p2_x = window_shape[0]
        p1_y += int(window_shape[1] * step)
        p2_y += int(window_shape[1] * step)

    x_data.shape = (i, window_shape[0], window_shape[1], window_shape[2])
    x_coord.shape = (i, 4)
    return {"x_data": x_data, "x_coord": x_coord, "longitudes": longitudes, "latitudes": latitudes}, \
           full_img, draw_image


def get_data_from_json_list(json_list, ex_shape, class_num):
    test_num = 0
    x_train = np.empty(0, dtype='uint8')
    y_train = np.empty(0, dtype='uint8')
    class_1_num = class_2_num = 0
    img_shape = (0, 0, 0)
    for train_json in json_list:
        cur_class_1_num, cur_class_2_num, img_shape, curr_x_train, curr_y_train = json_train_load(train_json)
        x_train = np.append(x_train, curr_x_train)
        y_train = np.append(y_train, curr_y_train)
        test_num += curr_y_train.shape[0]
        class_1_num += cur_class_1_num
        class_2_num += cur_class_2_num
    x_train.shape = (test_num, ex_shape[0], ex_shape[1], ex_shape[2])
    y_train.shape = (test_num, class_num)

    return class_1_num, class_2_num, img_shape, x_train, y_train


def json_train_create(path, cropped_data, y_data, img_shape, class_nums):
    if cropped_data["x_data"].shape[0] != y_data.shape[0]:
        raise Exception("bad shape")
    with open(path, "w") as fp:
        json.dump(
            {"class_nums": list(map(int, class_nums)), "img_shape": img_shape,
             "x_data": cropped_data["x_data"].tolist(), "y_data": y_data.tolist(),
             "longitudes": cropped_data["longitudes"], "latitudes": cropped_data["latitudes"]}, fp)
        fp.close()

    pass


def json_train_load(path):
    with open(path, "r") as fp:
        data_dict = json.load(fp)
        fp.close()
        return data_dict.get("class_1_num"), data_dict.get("class_2_num"), data_dict.get("img_shape"), \
               np.array(data_dict.get("x_data")), np.array(data_dict.get("y_data"))


################################################################################
# --------------------------------- multiple class -----------------------------
################################################################################
def multiple_class_examples(x_train, y_train, class_for_multiple,
                            use_noise=False, intensity_noise_list=(50,), use_deform=False, k_deform_list=(0.5,),
                            max_class_num=None):
    class_for_multiple_examples = np.empty(0)
    i = 0
    class_2_num = 0
    for y in y_train:
        if ((y.__eq__(class_for_multiple)).all()):
            class_for_multiple_examples = np.append(class_for_multiple_examples, x_train[i])
            class_2_num += 1
        i += 1

    class_for_multiple_examples.shape = (class_2_num, x_train.shape[1], x_train.shape[2], x_train.shape[3])

    class_1_num = y_train.shape[0] - class_2_num
    class_multiplied_result = np.empty(0)
    class_multiplayer = 1
    # --------------- make noised examples ----------------------------------
    if use_noise:
        for intensity in intensity_noise_list:
            class_multiplayer += 1
            for multiple_ex in class_for_multiple_examples:
                class_multiplied_result = \
                    np.append(class_multiplied_result, img_pr.noise_arr(arr=multiple_ex.flatten(), intensity=intensity))

    # --------------- make deformed examples ----------------------------------
    if use_deform:
        for k in k_deform_list:
            class_multiplayer += 1
            for multiple_ex in class_for_multiple_examples:
                class_multiplied_result = \
                    np.append(class_multiplied_result, img_pr.deform_arr(arr=multiple_ex, k=k, n=0, m=x_train.shape[1]))

    # ---------------  join arrays -------------------------------------------
    x_train = np.append(x_train, class_multiplied_result)
    for i in range(0, class_2_num * (class_multiplayer - 1)):
        y_train = np.append(y_train, class_for_multiple)

    x_train.shape = (class_1_num + class_2_num * class_multiplayer, 32, 32, 3)
    y_train.shape = (class_1_num + class_2_num * class_multiplayer, 2)

    if max_class_num is not None:
        if x_train.shape[0] > max_class_num:
            return x_train[0:max_class_num], y_train[0:max_class_num]

    return x_train, y_train


def get_pos_from_num(arr, class_num):
    new_arr = np.zeros((arr.size, class_num))

    curr_new_i = 0
    for num in arr:
        num = int(num)
        i = 0
        while num != 0:
            num -= 1
            i += 1
        new_arr[curr_new_i][i] = 1
        curr_new_i += 1
    return new_arr
