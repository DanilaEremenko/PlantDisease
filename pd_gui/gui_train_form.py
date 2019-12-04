from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget

from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_labels import TrainExLabel
from pd_gui.components.gui_colors import *
from pd_gui.components.gui_slider import ImgSizeSlider

from pd_lib.data_maker import get_x_from_croped_img, json_create
from pd_lib.img_proc import draw_rect_on_image

import numpy as np
import os


class WindowClassificationPicture(QWidget):
    def __init__(self):
        super(WindowClassificationPicture, self).__init__()
        self.hbox_image_list = []
        self.label_list = []
        self.setWindowTitle("Plant Disease Recognizer")
        self.img_path = self.choose_picture()
        self.img_name = os.path.splitext(self.img_path)[0]
        self.window_shape = (32, 32, 3)
        self.sl_min_val = 640
        self.sl_max_val = 1280
        self.img_shape = (self.sl_min_val, self.sl_min_val)

        self.button_init()
        self.update()
        self.show()
        pass

    def button_init(self):

        self.hbox_control = QtWidgets.QHBoxLayout()
        self.hbox_control.addStretch(1)
        self.sl = ImgSizeSlider(min_val=self.sl_min_val, max_val=self.sl_max_val, step_num=4, orientation='horizontal')
        self.hbox_control.addWidget(self.sl)
        self.hbox_control.addWidget(ControlButton("Okay", self.okay_pressed))
        self.hbox_control.addWidget(ControlButton("Update", self.update))
        self.hbox_control.addWidget(ControlButton("Quit", self.quit_pressed))

        self.main_box = QtWidgets.QVBoxLayout()
        self.main_box.addStretch(1)

        self.main_box.setContentsMargins(0, 0, 0, 0)
        self.main_box.setSpacing(0)
        self.setLayout(self.main_box)

    def clear(self):
        for hbox in self.hbox_image_list:
            hbox.setParent(None)
        for label in self.label_list:
            label.setParent(None)
        self.hbox_control.setParent(None)
        self.hbox_image_list = []
        self.label_list = []

    def update(self):
        self.clear()

        self.img_shape = (self.sl.value(), self.sl.value())
        self.x_data, self.x_coord, self.full_img, self.draw_image = get_x_from_croped_img(
            path_img_in=self.img_path,
            img_shape=self.img_shape,
            window_shape=self.window_shape,
            step=1.0,
            color=COLOR_GOOD
        )
        # -------------------- init image --------------------------
        x_len = int(self.full_img.size[0] / self.x_data.shape[1])
        y_len = int(self.full_img.size[1] / self.x_data.shape[2])
        i = 0
        for y in range(0, y_len):
            hbox_new = QtWidgets.QHBoxLayout()
            hbox_new.addStretch(1)
            for x in range(0, x_len):
                label_new = TrainExLabel(self.x_data[i].copy())
                hbox_new.addWidget(label_new)
                self.label_list.append(label_new)
                i += 1
            self.hbox_image_list.append(hbox_new)

        # -------------------- add boxes --------------------------
        for hbox in self.hbox_image_list:
            self.main_box.addLayout(hbox)
        self.main_box.addLayout(self.hbox_control)
        print("image updated")

    def choose_picture(self):
        return str(
            QtWidgets.QFileDialog.getOpenFileName(self, "Open *.png, *.jpg file with potato field", None,
                                                  "*.png *.PNG *.jpg *.JPG")[0])

    def okay_pressed(self):

        y_data = np.empty(0)
        i = 0
        class_1_num = class_2_num = 0
        for label in self.label_list:
            if label.type == 0:
                y_data = np.append(y_data, [0, 1])
                class_1_num += 1
                self.draw_image = draw_rect_on_image(self.draw_image, self.x_coord[i], color=COLOR_GOOD)
            elif label.type == 1:
                y_data = np.append(y_data, [1, 0])
                class_2_num += 1
                self.draw_image = draw_rect_on_image(self.draw_image, self.x_coord[i], color=COLOR_BAD)
            i += 1

        y_data.shape = (self.x_data.shape[0], 2)

        json_create(
            path="%s.json" % self.img_name,
            x_data=self.x_data,
            y_data=y_data,
            img_shape=self.full_img.size,
            class_1_num=class_1_num,
            class_2_num=class_2_num
        )

        self.draw_image.save("%s_net.JPG" % self.img_name)

        print("class_1_num = %d, class_2_num = %d" % (class_1_num, class_2_num))

        print("OKAY")

        self.quit_pressed()

        pass

    def quit_pressed(self):
        QtCore.QCoreApplication.instance().quit()
        pass
