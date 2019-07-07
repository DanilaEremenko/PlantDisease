from PIL import Image
from keras.preprocessing import image as kimage



def get_pxs(path):
    return kimage.img_to_array(kimage.load_img(path, color_mode="grayscale"))


def get_pxs_full(path):
    return kimage.img_to_array(kimage.load_img(path))