from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.preprocessing import image as kimage
import os


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


def plot_image_from_path(path):
    imshow(np.asarray(Image.open(path, 'r')))
    pass


def plot_image_from_arr(arr):
    im = Image.fromarray(arr)
    imshow(im)
    plt.show()
    pass
