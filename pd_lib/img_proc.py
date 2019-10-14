from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.preprocessing import image as kimage
import os


#############################################################################
# --------------------------- deform image ----------------------------------
#############################################################################
def deform_image(img, k, n, m):
    arr = np.array(img)
    if n > m:
        c = n
        n = m
        m = c
    A = arr.shape[0] / 3.0
    w = 2.0 / arr.shape[1]
    shift = lambda x: A * np.sin(2.0 * np.pi * x * w)
    for i in range(n, m):
        arr[:, i] = np.roll(arr[:, i], int(shift(i) * k))
    return Image.fromarray(arr)


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
# --------------------------- 'decropping' ----------------------------------
#############################################################################

def get_full_image_from_pieces(x_data, img_shape):
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


#############################################################################
# --------------------------- localizing ---------------------------------------
#############################################################################
def draw_rect(img, points, color):
    if points.__len__() != 4:
        raise Exception("points.__len__()!=4")
    px1, py1, px2, py2 = points[0:4]
    px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
    img_arr = np.array(img)
    # img_arr.setflags(write=1)
    img_arr[py1:py1 + 1, px1:px2] = color  # top
    img_arr[py2:py2 + 1, px1:px2] = color  # bottom
    img_arr[py1:py2, px1:px1 + 1] = color  # left
    img_arr[py1:py2, px2:px2 + 1] = color  # left
    return Image.fromarray(img_arr)


#############################################################################
# --------------------------- getters ---------------------------------------
#############################################################################
def get_pxs(path):
    return kimage.img_to_array(kimage.load_img(path, color_mode="grayscale"))


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
