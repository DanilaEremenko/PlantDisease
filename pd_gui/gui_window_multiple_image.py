from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget

from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_labels import ImageTextLabel

from pd_lib.data_maker import multiple_class_examples, json_train_create
from pd_lib.img_proc import get_full_rect_image_from_pieces

import json
import os
import numpy as np

intensity_noise_list = (50, 150)
k_deform_list = (0.09, 0.10, 0.11, 0.12, 0.13, 0.14)


class WindowMultipleExamples(QWidget):
    def __init__(self):
        super(WindowMultipleExamples, self).__init__()

        self.img_thumb_size = (768, 768)
        self.setWindowTitle("Plant Disease Recognizer")

        json_for_multiple = self.choose_json()
        self.init_data_from_json(json_for_multiple)

        self.define_max_class()

        self.json_name = os.path.splitext(json_for_multiple)[0]

        self.button_init()

        self.label_list = []
        self.hbox_image_list = []

        self.update_img()
        self.show()

    def init_data_from_json(self, json_for_multiple):
        with open(json_for_multiple) as train_json_fp:
            train_json = dict(json.load(train_json_fp))
            self.full_img_size = train_json['img_shape']
            self.classes = train_json['classes']
            self.x_train = np.array(train_json["x_data"], dtype='uint8')
            self.y_train = np.array(train_json["y_data"])
            self.longitudes, self.latitudes = train_json["longitudes"], train_json["latitudes"]
            self.img_shape = train_json["img_shape"]

    def define_max_class(self):
        self.max_class = {'name': None, 'num': 0, 'value': None}
        for key, value in self.classes.items():
            if self.classes[key]['num'] > self.max_class['num']:
                self.max_class['name'] = key
                self.max_class['num'] = self.classes[key]['num']
                self.max_class['value'] = self.classes[key]['value']

    def button_init(self):
        hbox_control = QtWidgets.QHBoxLayout()
        hbox_control.addStretch(1)
        hbox_control.addWidget(ControlButton("Okay", self.okay_pressed))
        hbox_control.addWidget(ControlButton("Multiple", self.multiple_pressed))
        hbox_control.addWidget(ControlButton("Quit", self.quit_pressed))

    def update_img(self):
        x_len = int(self.full_img_size[0] / self.x_train.shape[1])
        y_len = int(self.full_img_size[1] / self.x_train.shape[2])
        i = 0
        for y in range(0, y_len):
            hbox_new = QtWidgets.QHBoxLayout()
            hbox_new.addStretch(1)
            for x in range(0, x_len):
                label_new = ImageTextLabel(
                    self.x_train[i].copy(),
                    text='text'
                )
                hbox_new.addWidget(label_new)
                self.label_list.append(label_new)
                i += 1
            self.hbox_image_list.append(hbox_new)

        # -------------------- add boxes --------------------------
        for hbox in self.hbox_image_list:
            self.main_box.addLayout(hbox)
        self.main_box.addLayout(self.hbox_control)
        print("image updated")

    def choose_json(self):
        return str(
            QtWidgets.QFileDialog.getOpenFileName(self, "Open *.json file with train data field", None,
                                                  "*.json *.JSON")[0])

    def okay_pressed(self):
        out_json_path = "%s_multiple.json" % self.json_name
        print("class_1_num = %d, class_2_num = %d" % (self.class_1_num, self.class_2_num))
        print("Save to %s" % out_json_path)
        json_train_create(path=out_json_path,
                          cropped_data=
                          {"x_data": self.x_train, "longitudes": self.longitudes, "latitudes": self.latitudes},
                          y_data=self.y_train,
                          img_shape=None,
                          classes=self.classes)

        self.quit_pressed()

        pass

    def multiple_pressed(self):
        for key, value in self.classes.items():
            if (self.class_1_num + self.class_2_num) < self.max_class_num:
                self.x_train, self.y_train = multiple_class_examples(x_train=self.x_train, y_train=self.y_train,
                                                                     class_for_multiple=self.classes[key],
                                                                     use_noise=False,
                                                                     intensity_noise_list=intensity_noise_list,
                                                                     use_deform=True, k_deform_list=k_deform_list,
                                                                     max_class_num=self.max_class_num)

            self.update_img()
        else:
            print("class_1_num = %d, class_2_num = %d" % (self.class_1_num, self.class_2_num))
            print("max_class_num = %d" % self.max_class_num)

    def quit_pressed(self):
        QtCore.QCoreApplication.instance().quit()
        pass
