from pd_lib.addition import train_on_json
from pd_lib.conv_network import get_CNN
import os
from pd_lib.addition import get_full_model, predict_and_localize_on_image
import pd_lib.data_maker as dmk

if __name__ == '__main__':
    path_img_in = 'Datasets/PotatoFields/plan_train/DJI_0246.JPG'

    window_shape = (32, 32, 3)
    img_shape = (768, 768)

    model = get_full_model(json_path='models/model_ground_5.json', h5_path='models/model_ground_5.h5')

    x_data, x_coord, full_img, draw_image = \
        dmk.get_x_from_croped_img(path_img_in=path_img_in,
                                  img_shape=img_shape, window_shape=window_shape,
                                  step=1.0, color=255,
                                  path_out_dir=None)

    res_image = predict_and_localize_on_image(model=model, x_data=x_data, x_coord=x_coord, image=full_img)

    full_img.show()
    res_image.show()
