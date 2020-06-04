"""
Contains functions for train data formation and processing
"""
import h5py
import numpy as np
import os
from PIL import Image
from sklearn.utils import class_weight

from pd_lib import img_proc as img_pr, ui_cmd
from pd_geo import exif_parser as ep
import json

################################################################################
# --------------------------------- for predict on image -----------------------
################################################################################
def get_x_from_croped_img(path_to_img, window_shape, img_thumb=None, step=1.0, path_out_dir=None, verbose=False):
    if path_out_dir != None and not os.path.isdir(path_out_dir):
        raise Exception("No such directory %s" % path_out_dir)

    full_img = Image.open(path_to_img)
    if img_thumb is not None:
        full_img.thumbnail(img_thumb)

    img_exif = ep.get_exif_data(full_img)

    x_len = int(full_img.size[0] / window_shape[0])
    y_len = int(full_img.size[1] / window_shape[1])
    ex_num = x_len * y_len

    x_data = np.empty([ex_num, *window_shape[0:-1], 3], dtype='uint8')
    x_coord = []
    longitudes = []  # TODO add caluclating
    latitudes = []  # TODO add caluclating
    for i in range(ex_num):
        p1_x, p1_y = i % x_len * int(window_shape[0] * step), int(i / x_len) * window_shape[1]
        p2_x, p2_y = p1_x + int(window_shape[0] * step), p1_y + window_shape[1]

        if verbose:
            ui_cmd.printProgressBar(current=i, total=ex_num)
        if path_out_dir != None:
            img_pr.crop_multiply_data(img=full_img,
                                      name="%d" % i,
                                      crop_area=(p1_x, p1_y, p2_x, p2_y),
                                      path_out_dir=path_out_dir
                                      )

        x_data[i] = np.asarray(full_img.crop((p1_x, p1_y, p2_x, p2_y)))

    return {"x_data": x_data, "x_coord": x_coord, "longitudes": longitudes, "latitudes": latitudes}, \
           full_img


def get_data_from_json_list(json_list, remove_classes=None):
    test_num = 0
    x_train = np.empty(0, dtype='uint8')
    y_train = np.empty(0, dtype='uint8')
    classes = None
    ex_shape = None
    for train_json in json_list:
        with open(train_json) as json_fp:
            train_json = json.load(json_fp)
            curr_x_train = np.array(train_json["x_data"], dtype='uint8')
            curr_y_train = np.array(train_json["y_data"])
            cur_classes = {}

        # remove nesting
        for class_name in train_json['classes'].keys():
            for sub_class_name in train_json['classes'][class_name]:
                if remove_classes is not None \
                        and sub_class_name in remove_classes \
                        and train_json['classes'][class_name][sub_class_name]['num'] != 0:

                    remove_value = train_json['classes'][class_name][sub_class_name]['value'][0]
                    remove_indexes = []
                    for i, y in enumerate(curr_y_train):
                        if y[0] == remove_value:
                            remove_indexes.append(i)

                    new_curr_x_train = np.empty(0)
                    for i, x in enumerate(curr_x_train):
                        if i not in remove_indexes:
                            new_curr_x_train = np.append(new_curr_x_train, curr_x_train[i])

                    curr_x_train = np.array(new_curr_x_train, dtype='uint8').reshape(
                        (curr_x_train.shape[0] - len(remove_indexes), *curr_x_train.shape[1:]))
                    curr_y_train = np.delete(curr_y_train, remove_indexes)
                    curr_y_train.shape = (curr_y_train.shape[0], 1)

                    train_json['classes'][class_name][sub_class_name]['num'] = 0

                    print("'%s' examples removed" % sub_class_name)

                cur_classes[sub_class_name] = train_json['classes'][class_name][sub_class_name]
        # join with other data
        if ex_shape == None:
            ex_shape = curr_x_train.shape[1:]
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
    new_classes = {}
    for key in classes.keys():
        if classes[key]['num'] != 0:
            new_classes[key] = classes[key]
    classes = new_classes

    if len(classes.keys()) < 2:
        raise Exception('Illegal number of classes < 2')

    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y_train),
                                                      y=y_train)

    x_train.shape = (test_num, *ex_shape)
    y_values = set(y_train)
    map_dict = dict(zip(y_values, list(range(len(y_values)))))
    y_train = list(map(lambda x: map_dict[x], y_train))
    y_train = get_pos_from_num(arr=y_train, class_num=len(classes.keys()))
    for i, key in enumerate(classes.keys()):
        classes[key]['value'] = get_pos_from_num([map_dict[classes[key]['value'][0]]], class_num=len(classes.keys()))
        classes[key]['weight'] = [map_dict[list(y_values)[i]], class_weights[i]]

    return classes, x_train, y_train


###############################################################
# ---------------- store x & y arrays in h5 files -------------
###############################################################
def json_train_create(path, x_data_full, y_data, img_shape, classes):
    if x_data_full["x_data"].shape[0] != y_data.shape[0]:
        raise Exception("bad shape")
    with open(path, "w") as fp:
        json.dump(
            {
                "classes": classes, "img_shape": img_shape,
                "x_data": x_data_full["x_data"].tolist(), "y_data": y_data.tolist(),
                "longitudes": x_data_full["longitudes"], "latitudes": x_data_full["latitudes"]
            },
            fp)
        fp.close()


def json_train_load(path):
    with open(path, "r") as fp:
        data_dict = json.load(fp)
        fp.close()
        return data_dict["classes"], data_dict["img_shape"], \
               np.array(data_dict["x_data"], dtype='uint8'), np.array(data_dict["y_data"], dtype='uint8')


###############################################################
# ---------------- store x & y arrays in h5 files -------------
###############################################################
def json_big_create(json_path, h5_path, x_data, y_data, img_shape, longitudes, latitudes, classes):
    h5f = h5py.File(h5_path, "w")
    h5f.create_dataset(name='x_data', data=x_data)
    h5f.create_dataset(name='y_data', data=y_data)
    h5f.close()
    with open(json_path, "w") as fp:
        json.dump(
            {
                "classes": classes, "img_shape": img_shape,
                "h5d_path": h5_path,
                "longitudes": longitudes, "latitudes": latitudes
            },
            fp)
        fp.close()


def json_big_load(json_path):
    with open(json_path, "r") as json_fp:
        config_dict = json.load(json_fp)
    h5f = h5py.File(config_dict["h5d_path"], 'r')
    x_data = np.array(h5f.get('x_data'))
    y_data = np.array(h5f.get('y_data'))
    h5f.close()
    return config_dict['classes'], config_dict['img_shape'], x_data, y_data


################################################################################
# --------------------------------- multiple class -----------------------------
################################################################################
def multiple_class_examples(x_train, y_train,
                            class_for_mult=None,
                            use_noise=False, intensity_noise_list=(50,),
                            use_deform=False, k_deform_list=(0.5,),
                            use_blur=False, rad_list=(0.5),
                            use_affine=False, affine_list=(0.5,),
                            use_contrast=False, contrast_list=(0.7,),
                            max_class_num=None,
                            mode='not all'):
    class_for_mult = np.array(class_for_mult)

    class_for_multiple_examples = np.empty(0, dtype='uint8')
    class_for_mult_num = 0

    if mode == 'all':
        class_for_multiple_examples = x_train
        class_for_mult = len(x_train)
    else:
        for i, y in enumerate(y_train):
            if ((y.__eq__(class_for_mult)).all()):
                class_for_multiple_examples = np.append(class_for_multiple_examples, x_train[i])
                class_for_mult_num += 1
        class_for_multiple_examples.shape = (class_for_mult_num, x_train.shape[1], x_train.shape[2], x_train.shape[3])

    algh_dict = {}
    # define alghoritms
    if use_noise:
        algh_dict['noised'] = {
            'func': img_pr.noise_arr,
            'args': {},
            'loop_list': intensity_noise_list,
            'loop_arg': 'intensity'
        }
    if use_deform:
        algh_dict['deformed'] = {
            'func': img_pr.deform_arr,
            'args': {'n': 0, 'm': x_train.shape[1]},
            'loop_list': k_deform_list,
            'loop_arg': 'k'
        }
    if use_blur:
        algh_dict['blured'] = {
            'func': img_pr.blur_arr,
            'args': {},
            'loop_list': rad_list,
            'loop_arg': 'radius'
        }
    if use_affine:
        algh_dict['affine'] = {
            'func': img_pr.affine_arr,
            'args': {},
            'loop_list': affine_list,
            'loop_arg': 'k'
        }
    if use_contrast:
        algh_dict['contrast'] = {
            'func': lambda arr, k: arr * k if k < 1 else np.vectorize(lambda x: min(255, x))(arr * k),
            'args': {},
            'loop_list': contrast_list,
            'loop_arg': 'k'
        }

    max_new_examples_num = max_class_num - class_for_mult_num
    x_new_examples = np.empty(0, dtype='uint8')

    # --------------- make noised examples ----------------------------------
    for alg_name, algh in algh_dict.items():
        print('generating via %s' % alg_name)
        for loop_var in algh['loop_list']:
            print('augmentation with k = %.2f' % loop_var)
            if len(x_new_examples) < max_new_examples_num:
                x_new_examples = np.append(
                    x_new_examples, np.array(
                        list(
                            map(
                                lambda mult_ex: algh['func'](**{'arr': mult_ex, algh['loop_arg']: loop_var},
                                                             **algh['args']),
                                class_for_multiple_examples
                            )
                        ),
                        dtype='uint8'
                    )
                )
                x_new_examples.shape = (
                    int(x_new_examples.shape[0] / (x_train.shape[1] * x_train.shape[2] * x_train.shape[3])),
                    *x_train.shape[1:]
                )

    # ---------------  join arrays -------------------------------------------
    if mode == 'all':
        mult_num = int(len(x_new_examples) / len(y_train))
        y_new_examples = np.array(list(y_train) * mult_num, dtype='uint8')

    else:
        y_new_examples = np.array(list(class_for_mult.astype('uint8')) * len(x_new_examples)) \
            .reshape((len(x_new_examples), len(class_for_mult)))

    return x_new_examples, y_new_examples


def get_pos_from_num(arr, class_num):
    new_arr = np.zeros((len(arr), class_num))

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


def get_num_from_pos(arr):
    for i, n in enumerate(arr):
        if n == 1:
            return i
    raise Exception("%s is not a pos code" % str(arr))
