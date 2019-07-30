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


def crop_img_window(path_img_in, img_shape, window_shape, path_out_dir, draw_net=True, make_test_dir=False):
    if not os.path.isdir(path_out_dir):
        raise Exception("No such directory %s" % path_out_dir)

    full_img = Image.open(path_img_in)  # TODO
    full_img.thumbnail(img_shape)
    full_img.show()
    p1_x, p1_y, p2_x, p2_y = 0, 0, window_shape[0], window_shape[1]
    i = 0
    while p2_y <= full_img.size[1]:
        while p2_x <= full_img.size[0]:
            if make_test_dir:
                crop_multiply_data(img=full_img,
                                   name="%d" % i,
                                   crop_area=(p1_x, p1_y, p2_x, p2_y),
                                   path_out_dir=path_out_dir
                                   )

            if draw_net: full_img = draw_rect(full_img, (p1_x, p1_y, p2_x, p2_y), color=255)

            p1_x += window_shape[0]
            p2_x += window_shape[0]
            i += 1
        p1_x = 0
        p2_x = window_shape[0]
        p1_y += window_shape[1]
        p2_y += window_shape[1]
    if draw_net: full_img.show()
    pass


#############################################################################
# --------------------------- localizing ---------------------------------------
#############################################################################
def draw_rect(img, points, color):
    if points.__len__() != 4:
        raise Exception("points.__len__()!=4")
    px1, py1, px2, py2 = points[0:4]
    img_arr = np.asarray(img)
    img_arr.setflags(write=1)
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
