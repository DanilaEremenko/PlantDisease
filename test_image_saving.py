from pd_lib.data_maker import json_load
from pd_lib.img_proc import plot_full_from_multiple

x_train, y_train, img_shape = json_load('Datasets/PotatoFields/plan_train/DJI_0246.json')
plot_full_from_multiple(x_data=x_train, img_shape=img_shape)
