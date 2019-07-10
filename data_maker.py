import numpy as np
import os
from img_proc import get_pxs_full


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


def get_data(data, img_shape, max_img_num=None):
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
