import numpy as np
import os
from img_proc import get_pxs_full


def get_classes_from_dir(path_to_dir, class_mark, max_img_num=None):
    curr_x = np.empty(0)
    curr_y = np.empty(0)
    i = 0

    for path_to_img in os.listdir(path_to_dir):
        curr_x = np.append(curr_x, get_pxs_full("%s/%s" % (path_to_dir, path_to_img)))
        curr_y = np.append(curr_y, class_mark)
        i += 1

        if i == 1:
            img_shape = get_pxs_full("%s/%s" % (path_to_dir, path_to_img)).shape
        if max_img_num != None:
            if i == max_img_num:
                break

    curr_x.shape = (i, img_shape[0], img_shape[1], img_shape[2])

    return curr_x, curr_y


def get_data(paths_to_dirs, class_marks, max_img_num=None):
    if paths_to_dirs.__len__() != class_marks.__len__():
        raise Exception("paths_to_dirs and class_marks must have the same size")

    x_train = np.empty(0)
    y_train = np.empty(0)
    for path_to_dir, class_mark in zip(paths_to_dirs, class_marks):
        (curr_x, curr_y) = get_classes_from_dir(path_to_dir, class_mark, max_img_num)
        x_train = np.append(x_train, curr_x)
        y_train = np.append(y_train, curr_y)

    return x_train, y_train
