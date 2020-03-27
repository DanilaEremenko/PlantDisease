"""
Contains functions for image processing
"""
import cv2
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageFilter


#############################################################################
# --------------------------- deform image ----------------------------------
#############################################################################
def get_img_edges(img, thr_1=60, thr_2=120):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 20, 30)
    edges_high_thresh = cv2.Canny(gray, thr_1, thr_2)
    return edges_high_thresh


#############################################################################
# --------------------------- deform image ----------------------------------
#############################################################################
def deform_image(img, k, n, m):
    return Image.fromarray(deform_arr(np.asarray(img), k, n, m))


def deform_arr(arr, k, n, m):
    """
    :param arr: img as array
    :param k: intensity
    :param n: from pixel
    :param m: to pixel
    :return: deformed image
    """
    res_arr = arr.copy()
    if n > m:
        c = n
        n = m
        m = c
    A = res_arr.shape[0] / 3.0
    w = 2.0 / res_arr.shape[1]
    shift = lambda x: A * np.sin(2.0 * np.pi * x * w)
    for i in range(n, m):
        res_arr[:, i] = np.roll(res_arr[:, i], int(shift(i) * k))

    return res_arr


#############################################################################
# --------------------------- noise image -----------------------------------
#############################################################################
def noise_img_from_arr(img, intensity):
    return Image.fromarray(noise_arr(np.asarray(img), intensity))


def noise_arr(arr, intensity):
    res_arr = arr.copy().flatten()
    for i in range(0, res_arr.size):
        res_arr[i] = (res_arr[i] + np.random.randint(0, intensity)) % 255
    return res_arr


#############################################################################
# --------------------------- blur image -----------------------------------
#############################################################################
def blur_img(arr, radius):
    img = Image.fromarray(arr)
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=radius)))


#############################################################################
# --------------------------- warp image ------------------------------------
#############################################################################
def affine_warp(arr, k):
    y, x = arr.shape[0:2]
    step = max(1, int(x * k / 100))

    pts1 = np.float32([[step, step], [x, step], [step, y]])
    pts2 = np.float32(
        [
            [0.5 * step, 1.5 * step],
            [x - 0.5 * step, 0.5 * step],
            [1.5 * step, y - 0.5 * step]
        ]
    )

    M = cv2.getAffineTransform(pts1, pts2)

    return cv2.warpAffine(arr, M, (x, y))


#############################################################################
# --------------------------- cropping --------------------------------------
#############################################################################
def crop_img(img, crop_area):
    return img.crop(crop_area)


def get_all_perspectives(img):
    return img.rotate(0), img.rotate(90), img.rotate(180), img.rotate(270)


def crop_multiply_data(img, name, crop_area, path_out_dir):
    for img, angle in zip(get_all_perspectives(crop_img(img=img, crop_area=crop_area)),
                          (0, 90, 180, 270)):
        img.save("%s/%s_%d.jpg" % (path_out_dir, name, angle))
    pass


#############################################################################
# --------------------------- merging ---------------------------------------
#############################################################################
def get_full_repaired_image_from_pieces(x_data, img_shape):
    x_len = int(img_shape[0] / x_data.shape[1])
    y_len = int(img_shape[1] / x_data.shape[2])

    img_shape = (x_len * x_data.shape[1],
                 y_len * x_data.shape[2],
                 3)

    res_image = np.empty(img_shape, dtype='uint8')

    window_shape = x_data[0].shape
    i = 0
    for x in range(0, x_len):
        for y in range(0, y_len):
            res_image[x * window_shape[0]:(x + 1) * window_shape[1], y * window_shape[1]:(y + 1) * window_shape[1]] = \
                x_data[i]
            i += 1
    img = Image.fromarray(res_image, mode='RGB')
    return img


def get_full_rect_image_from_pieces(x_data, color_mode='RGB'):
    rect_size = int(np.sqrt(x_data.shape[0]) + 1)

    if color_mode == 'RGB':
        res_image = np.empty((x_data.shape[1] * rect_size, x_data.shape[2] * rect_size, x_data.shape[3]), dtype='uint8')
    elif color_mode == 'L':
        res_image = np.empty((x_data.shape[1] * rect_size, x_data.shape[2] * rect_size), dtype='uint8')
    else:
        raise Exception("Undefined color_mode = %s" % color_mode)

    window_shape = x_data[0].shape
    i = 0
    for x in range(0, rect_size):
        for y in range(0, rect_size):
            res_image[x * window_shape[0]:(x + 1) * window_shape[1], y * window_shape[1]:(y + 1) * window_shape[1]] = \
                x_data[i]
            i += 1
            if i == x_data.shape[0]:
                return Image.fromarray(res_image, mode=color_mode)
    return Image.fromarray(res_image, mode=color_mode)


#############################################################################
# --------------------------- drawing ---------------------------------------
#############################################################################
def draw_rect_on_image(img, points, color):
    return Image.fromarray(draw_rect_on_array(img_arr=np.array(img), points=points, color=color))


def draw_rect_on_array(img_arr, points, color):
    if points.__len__() != 4:
        raise Exception("points.__len__()!=4")
    px1, py1, px2, py2 = points[0:4]
    px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)

    img_arr[py1:py1 + 1, px1:px2] = color  # top
    img_arr[py2:py2 + 1, px1:px2] = color  # bottom
    img_arr[py1:py2, px1:px1 + 1] = color  # left
    img_arr[py1:py2, px2:px2 + 1] = color  # left
    return img_arr


#############################################################################
# --------------------------- getters ---------------------------------------
#############################################################################
# def get_pxs(path):
#     return kimage.img_to_array(kimage.load_img(path, color_mode="grayscale"))


def get_pxs_full(path, shape=None):
    if shape == None:
        return np.asarray(Image.open(path))
    else:
        im = Image.open(path)
        im.thumbnail(shape)
        return np.asarray(im)


def get_imgs_shapes(path_to_dir):
    return get_pxs_full("%s/%s" % (path_to_dir, os.listdir(path_to_dir)[0])).shape


#############################################################################
# ------------------------------- plotting ----------------------------------
#############################################################################
def plot_image_from_path(path):
    imshow(np.asarray(Image.open(path, 'r')))
    pass


def plot_image_from_arr(arr):
    im = Image.fromarray(arr)
    imshow(im)
    plt.show()
    pass
