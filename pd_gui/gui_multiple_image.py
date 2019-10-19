from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget

from .gui_buttons import ControlButton
from .gui_labels import ImageLabel

from pd_lib.data_maker import multiple_class_examples, json_create
from pd_lib.img_proc import get_full_rect_image_from_pieces, draw_rect_on_array
from .gui_colors import COLOR_GOOD

import json
import os
import numpy as np

intensity_noise_list = (50, 150)
k_deform_list = (0.09, 0.10, 0.11, 0.12, 0.13, 0.14)


class WindowMultipleExamples(QWidget):
    def __init__(self):
        super(WindowMultipleExamples, self).__init__()
        self.setWindowTitle("Plant Disease Recognizer")
        self.setMouseTracking(True)
        json_for_multiple = self.choose_json()

        with open(json_for_multiple) as train_json_fp:
            train_json = dict(json.load(train_json_fp))
            self.class_1_num, self.class_2_num, self.x_train, self.y_train, img_shape = \
                train_json.get("class_1_num"), train_json.get("class_2_num"), \
                np.array(train_json.get("x_data"), dtype='uint8'), \
                np.array(train_json.get("y_data")), \
                train_json.get("img_shape")

        if self.class_1_num < self.class_2_num:
            self.mult_class = (0, 1)
        elif self.class_2_num < self.class_1_num:
            self.mult_class = (1, 0)
        else:
            print("class_1_num == class_2_num, exiting...")
            self.quit_pressed()

        print(self.mult_class)
        self.max_class_num = max([self.class_1_num, self.class_2_num]) * 2
        self.json_name = os.path.splitext(json_for_multiple)[0]
        self.button_init()
        self.show()
        pass

    def button_init(self):
        hbox_control = QtWidgets.QHBoxLayout()
        hbox_control.addStretch(1)
        hbox_control.addWidget(ControlButton("Okay", self.okay_pressed))
        hbox_control.addWidget(ControlButton("Multiple", self.multiple_pressed))
        hbox_control.addWidget(ControlButton("Quit", self.quit_pressed))

        self.hbox_img = QtWidgets.QHBoxLayout()
        self.hbox_img.addStretch(1)
        self.img_label = ImageLabel(
            np.asarray(get_full_rect_image_from_pieces(x_data=self.get_marked_img(class_code=self.mult_class))))
        self.hbox_img.addWidget(self.img_label)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addStretch(1)

        vbox.addLayout(self.hbox_img)
        vbox.addLayout(hbox_control)

        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        self.setLayout(vbox)

    def choose_json(self):
        return str(
            QtWidgets.QFileDialog.getOpenFileName(self, "Open *.json file with train data field", None,
                                                  "*.json *.JSON")[0])

    def okay_pressed(self):
        out_json_path = "%s_multiple.json" % self.json_name
        print("class_1_num = %d, class_2_num = %d" % (self.class_1_num, self.class_2_num))
        print("Save to %s" % out_json_path)
        json_create(path=out_json_path,
                    x_data=self.x_train, y_data=self.y_train,
                    img_shape=None,
                    class_1_num=self.class_1_num, class_2_num=self.class_2_num)

        self.quit_pressed()

        pass

    def multiple_pressed(self):
        if (self.class_1_num + self.class_2_num) < self.max_class_num:
            self.x_train, self.y_train = multiple_class_examples(x_train=self.x_train, y_train=self.y_train,
                                                                 class_for_multiple=self.mult_class,
                                                                 use_noise=False,
                                                                 intensity_noise_list=intensity_noise_list,
                                                                 use_deform=True, k_deform_list=k_deform_list,
                                                                 max_class_num=max(
                                                                     [self.class_1_num, self.class_2_num]) * 2)

            self.class_2_num = self.y_train.shape[0] - self.class_1_num
            self.img_label.setParent(None)
            self.img_label = ImageLabel(
                np.asarray(get_full_rect_image_from_pieces(x_data=self.get_marked_img(class_code=self.mult_class))))
            self.hbox_img.addWidget(self.img_label)
        else:
            print("class_1_num = %d, class_2_num = %d" % (self.class_1_num, self.class_2_num))
            print("max_class_num = %d" % self.max_class_num)

    def quit_pressed(self):
        QtCore.QCoreApplication.instance().quit()
        pass

    def get_marked_img(self, class_code):
        x_drawable = self.x_train.copy()

        i = 0
        for y_ex in self.y_train:
            if y_ex.__eq__(class_code).all():
                x_drawable[i] = draw_rect_on_array(x_drawable[i],
                                                   points=(1, 1, x_drawable.shape[1] - 1, x_drawable.shape[2] - 1),
                                                   color=COLOR_GOOD)
            i += 1

        return x_drawable
