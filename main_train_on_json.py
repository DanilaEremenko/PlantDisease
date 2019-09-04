from pd_lib.addition import train_on_json
from pd_lib.conv_network import get_CNN
import os
from pd_lib.addition import save_to_json

if __name__ == '__main__':
    train_dir = 'Datasets/PotatoFields/plan_train'

    img_shape = (32, 32, 3)
    class_num = 2
    test_num = 0
    json_list = []
    #####################################################################
    # ----------------------- create train data -------------------------
    #####################################################################
    for train_json in os.listdir(train_dir):
        file_format = train_json.split(".")
        if file_format.__len__() == 2 and file_format[1] == 'json':
            json_list.append("%s/%s" % (train_dir, train_json))
    for train_json in json_list:
        print(train_json)
    #####################################################################
    # ----------------------- train_model -------------------------------
    #####################################################################
    verbose = True
    history_show = True
    title = 'train on ground'
    epochs = 5

    model = get_CNN(img_shape, class_num)
    model = train_on_json(model=model, json_list=json_list,
                          img_shape=img_shape, class_num=class_num,
                          epochs=epochs,
                          verbose=verbose, history_show=history_show)

    save_to_json(model, "models/model_ground_%d.json" % epochs)
    model.save_weights('models/model_ground_%d.h5' % epochs)
