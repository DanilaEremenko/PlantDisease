import numpy as np
import os
from PIL import Image
from pd_lib import img_proc as img_pr, ui_cmd
from pd_geo import exif_parser as ep
import json


################################################################################
# --------------------------------- for predict on image -----------------------
################################################################################
def get_x_from_croped_img(path_img_in, window_shape, img_thumb=None, step=1.0, color=255, path_out_dir=None,
                          verbose=False):
    if path_out_dir != None and not os.path.isdir(path_out_dir):
        raise Exception("No such directory %s" % path_out_dir)

    full_img = Image.open(path_img_in)
    if img_thumb is not None:
        full_img.thumbnail(img_thumb)

    img_exif = ep.get_exif_data(full_img)

    draw_image = full_img

    p1_x, p1_y, p2_x, p2_y = 0, 0, window_shape[0], window_shape[1]
    i = 0
    ex_num = int(full_img.size[0] / window_shape[0]) * int(full_img.size[1] / window_shape[1])
    x_data = np.empty([ex_num, *window_shape[0:-1], 3], dtype='uint8')
    x_coord = []
    longitudes = []  # TODO add caluclating
    latitudes = []  # TODO add caluclating
    while p2_y <= full_img.size[1]:
        while p2_x <= full_img.size[0]:
            if verbose:
                ui_cmd.printProgressBar(current=i, total=ex_num)
            if path_out_dir != None:
                img_pr.crop_multiply_data(img=full_img,
                                          name="%d" % i,
                                          crop_area=(p1_x, p1_y, p2_x, p2_y),
                                          path_out_dir=path_out_dir
                                          )

            x_data[i] = np.asarray(full_img.crop((p1_x, p1_y, p2_x, p2_y)))

            draw_image = img_pr.draw_rect_on_image(draw_image, (p1_x, p1_y, p2_x, p2_y), color=color)

            p1_x += int(window_shape[0] * step)
            p2_x += int(window_shape[0] * step)
            i += 1
        p1_x = 0
        p2_x = window_shape[0]
        p1_y += int(window_shape[1] * step)
        p2_y += int(window_shape[1] * step)
    return {"x_data": x_data, "x_coord": x_coord, "longitudes": longitudes, "latitudes": latitudes}, \
           full_img, draw_image


def get_data_from_json_list(json_list, ex_shape):
    test_num = 0
    x_train = np.empty(0, dtype='uint8')
    y_train = np.empty(0, dtype='uint8')
    classes = None
    for train_json in json_list:
        cur_classes, img_shape, curr_x_train, curr_y_train = json_train_load(train_json)
        if classes is None:
            classes = cur_classes
        else:
            for key in classes.keys():
                if classes[key]['value'] == cur_classes[key]['value']:
                    classes[key]['num'] += cur_classes[key]['num']
                else:
                    raise Exception("cur_classes.key.value = %s" % str(cur_classes[key]['value']))
        x_train = np.append(x_train, curr_x_train)
        y_train = np.append(y_train, curr_y_train)
        test_num += curr_y_train.shape[0]
    x_train.shape = (test_num, ex_shape[0], ex_shape[1], ex_shape[2])
    y_train.shape = (test_num, len(classes))

    return classes, x_train, y_train


def json_train_create(path, x_data_full, y_data, img_shape, classes):
    if x_data_full["x_data"].shape[0] != y_data.shape[0]:
        raise Exception("bad shape")
    with open(path, "w") as fp:
        json.dump(
            {"classes": classes, "img_shape": img_shape,
             "x_data": x_data_full["x_data"].tolist(), "y_data": y_data.tolist(),
             "longitudes": x_data_full["longitudes"], "latitudes": x_data_full["latitudes"]}, fp)
        fp.close()

    pass


def json_train_load(path):
    with open(path, "r") as fp:
        data_dict = json.load(fp)
        fp.close()
        return data_dict["classes"], data_dict["img_shape"], \
               np.array(data_dict["x_data"]), np.array(data_dict["y_data"])


################################################################################
# --------------------------------- multiple class -----------------------------
################################################################################
def multiple_class_examples(x_train, y_train, class_for_multiple,
                            use_noise=False, intensity_noise_list=(50,), use_deform=False, k_deform_list=(0.5,),
                            max_class_num=None):
    original_len = len(y_train)

    class_for_multiple_examples = np.empty(0, dtype='uint8')
    class_for_mult_num = 0

    for i, y in enumerate(y_train):
        if ((y.__eq__(class_for_multiple)).all()):
            class_for_multiple_examples = np.append(class_for_multiple_examples, x_train[i])
            class_for_mult_num += 1

    class_for_multiple_examples.shape = (class_for_mult_num, x_train.shape[1], x_train.shape[2], x_train.shape[3])

    max_new_examples_num = max_class_num - class_for_mult_num
    new_examples_num = 0

    x_new_examples = np.empty(0, dtype='uint8')

    stop_augment = False
    # --------------- make noised examples ----------------------------------
    if use_noise:
        for intensity in intensity_noise_list:
            if stop_augment:
                break
            for multiple_ex in class_for_multiple_examples:
                if new_examples_num < max_new_examples_num:
                    x_new_examples = \
                        np.append(x_new_examples, img_pr.noise_arr(arr=multiple_ex.flatten(), intensity=intensity))
                    new_examples_num += 1
                else:
                    stop_augment = True
                    break
    # --------------- make deformed examples ----------------------------------
    if use_deform:
        for k in k_deform_list:
            if stop_augment:
                break
            for multiple_ex in class_for_multiple_examples:
                if new_examples_num < max_new_examples_num:
                    x_new_examples = \
                        np.append(x_new_examples, img_pr.deform_arr(arr=multiple_ex, k=k, n=0, m=x_train.shape[1]))
                    new_examples_num += 1
                else:
                    stop_augment = True
                    break

    # ---------------  join arrays -------------------------------------------
    x_train = np.append(x_train, x_new_examples)
    for i in range(new_examples_num):
        y_train = np.append(y_train, class_for_multiple)

    x_train.shape = (original_len + new_examples_num, 32, 32, 3)
    y_train.shape = (original_len + new_examples_num, len(class_for_multiple))

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
