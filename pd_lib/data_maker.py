import numpy as np
import os
from pd_lib.img_proc import get_pxs_full
from PIL import Image
from pd_lib.img_proc import crop_multiply_data, draw_rect
import json


################################################################################
# --------------------------------- for train & test on dirs -------------------
################################################################################
def get_class_from_dir(path_to_dir, class_mark, img_shape, max_img_num=None):
    curr_x = np.empty(0)
    curr_y = np.empty(0)
    i = 0

    for path_to_img in os.listdir(path_to_dir):
        img_pxs = get_pxs_full("%s/%s" % (path_to_dir, path_to_img), shape=img_shape)
        if img_pxs.shape == img_shape:
            curr_x = np.append(curr_x, get_pxs_full("%s/%s" % (path_to_dir, path_to_img), shape=img_shape))
            curr_y = np.append(curr_y, class_mark)
            i += 1

            if max_img_num != None:
                if i == max_img_num:
                    break

    curr_x.shape = (i, img_shape[0], img_shape[1], img_shape[2])

    return curr_x, curr_y, i


def get_data_full(data, img_shape, max_img_num=None):
    if not data.keys().__contains__("class_marks"):
        raise Exception("no class_marks key in data")
    if not data.keys().__contains__("data_dirs"):
        raise Exception("no data_dirs key in data")
    if data.get("class_marks").__len__() != data.get("data_dirs").__len__():
        raise Exception("class_marks and data_dirs must have the same size")

    x_train = np.empty(0)
    y_train = np.empty(0)
    im_sum = 0
    for path_to_dir, class_mark in zip(data.get("data_dirs"), data.get("class_marks")):
        (curr_x, curr_y, i) = get_class_from_dir(path_to_dir, class_mark, img_shape, max_img_num)
        x_train = np.append(x_train, curr_x)
        y_train = np.append(y_train, curr_y)
        im_sum += i
    x_train.shape = (im_sum, img_shape[0], img_shape[1], img_shape[2])
    y_train.shape = (im_sum, data.get("class_marks")[0].shape[0])
    return x_train, y_train


################################################################################
# --------------------------------- for predict on dir -------------------------
################################################################################
def get_x_from_dir(path_to_dir, img_shape, max_img_num=None):
    curr_x = np.empty(0)
    i = 0
    for path_to_img in os.listdir(path_to_dir):
        img_pxs = get_pxs_full("%s/%s" % (path_to_dir, path_to_img), shape=img_shape)
        if img_pxs.shape == img_shape:
            curr_x = np.append(curr_x, get_pxs_full("%s/%s" % (path_to_dir, path_to_img), shape=img_shape))
            i += 1

            if max_img_num != None:
                if i == max_img_num:
                    break

    curr_x.shape = (i, img_shape[0], img_shape[1], img_shape[2])

    return curr_x


################################################################################
# --------------------------------- for predict on image -----------------------
################################################################################
def get_x_from_croped_img(path_img_in, img_shape, window_shape, step=1.0, color=255, path_out_dir=None):
    if path_out_dir != None and not os.path.isdir(path_out_dir):
        raise Exception("No such directory %s" % path_out_dir)

    full_img = Image.open(path_img_in)
    full_img.thumbnail(img_shape)

    draw_image = full_img

    p1_x, p1_y, p2_x, p2_y = 0, 0, window_shape[0], window_shape[1]
    i = 0
    x_data = np.empty(0, dtype='uint8')
    x_coord = np.empty(0, dtype='uint8')

    while p2_y <= full_img.size[1]:
        while p2_x <= full_img.size[0]:
            if path_out_dir != None:
                crop_multiply_data(img=full_img,
                                   name="%d" % i,
                                   crop_area=(p1_x, p1_y, p2_x, p2_y),
                                   path_out_dir=path_out_dir
                                   )

            curr_x = np.asarray(full_img.crop((p1_x, p1_y, p2_x, p2_y)))
            x_data = np.append(x_data, curr_x)
            x_coord = np.append(x_coord, (p1_x, p1_y, p2_x, p2_y))

            draw_image = draw_rect(draw_image, (p1_x, p1_y, p2_x, p2_y), color=color)

            p1_x += int(window_shape[0] * step)
            p2_x += int(window_shape[0] * step)
            i += 1
        p1_x = 0
        p2_x = window_shape[0]
        p1_y += int(window_shape[1] * step)
        p2_y += int(window_shape[1] * step)

    x_data.shape = (i, window_shape[0], window_shape[1], window_shape[2])
    x_coord.shape = (i, 4)
    return x_data, x_coord, full_img, draw_image


def get_data_from_json_list(json_list, img_shape, class_num):
    test_num = 0
    x_train = np.empty(0)
    y_train = np.empty(0)
    for train_json in json_list:
        curr_x_train, curr_y_train = json_load(train_json)
        x_train = np.append(x_train, curr_x_train)
        y_train = np.append(y_train, curr_y_train)
        test_num += curr_y_train.shape[0]
    x_train.shape = (test_num, img_shape[0], img_shape[1], img_shape[2])
    y_train.shape = (test_num, class_num)

    return x_train, y_train


def json_create(path, x_data, y_data, img_shape):
    if x_data.shape[0] != y_data.shape[0]:
        raise Exception("bad shape")
    with open(path, "w") as fp:
        json.dump({"x_data": x_data.tolist(), "y_data": y_data.tolist(), "img_shape": img_shape}, fp)
        fp.close()

    pass


def json_load(path):
    with open(path, "r") as fp:
        data_dict = json.load(fp)
        fp.close()
        return np.array(data_dict.get("x_data")), np.array(data_dict.get("y_data")), data_dict.get("img_shape")
