from addition import get_full_model, predict_and_localize_on_image
from conv_network import get_CNN
from data_maker import get_x_from_croped_img

if __name__ == '__main__':
    path_img_in = "Datasets/PotatoFields/plan/DJI_0246.JPG"
    window_shape = (32, 32, 3)

    x_data, x_coord, full_img = get_x_from_croped_img(
        path_img_in=path_img_in,
        img_shape=(512, 512),
        window_shape=window_shape,
        path_out_dir="Datasets/PotatoFields/plan_processed",
        draw_net=True
    )

    # model = get_full_model(json_path="models/model_potato_30.json",
    #                        h5_path="models/model_potato_30.h5")

    model = get_CNN(img_shape=window_shape, out_neurons_num=2)

    res_image = predict_and_localize_on_image(model, x_data, x_coord, full_img)

    res_image.show()
