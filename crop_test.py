from img_proc import crop_img_window

if __name__ == '__main__':
    img_shape = (32, 32)
    crop_img_window(
        path_img_in="Datasets/PotatoFields/plan/DJI_0246.JPG",
        window_shape=img_shape,
        path_out_dir="Datasets/PotatoFields/plan_processed"
    )
