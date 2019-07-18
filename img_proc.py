from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.preprocessing import image as kimage
import os


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


def crop_img_window(path_img_in, window_shape, path_out_dir):
    full_img = Image.open(path_img_in)
    p1_x, p1_y, p2_x, p2_y = 0, 0, window_shape[0], window_shape[1]
    i = 0
    while p2_y <= full_img.size[1]:
        while p2_x <= full_img.size[0]:
            # crop_multiply_data(img=full_img,
            #                    name="%d" % name_i,
            #                    crop_area=(p1_x, p1_y, p2_x, p2_y),
            #                    path_out_dir=path_out_dir
            #                    )
            p1_x += window_shape[0]
            p2_x += window_shape[0]
            print "%d:p1_x = %d, p1_y = %d, p2_x = %d, p2_y = %d" % (i, p1_x, p1_y, p2_x, p2_y)
            i += 1
        p1_x = 0
        p2_x = window_shape[0]
        p1_y += window_shape[1]
        p2_y += window_shape[1]

    pass


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
