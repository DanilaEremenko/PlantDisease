from PyQt5 import QtWidgets

from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_labels import TrainExLabel
from pd_gui.components.gui_colors import *
from pd_gui.components.gui_slider import ImgSizeSlider
from pd_gui.components.gui_layouts import MyGridLayout
from pd_gui.gui_window_interface import WindowInterface

from pd_lib import data_maker as dmk

import numpy as np
import os


class WindowClassificationPicture(WindowInterface):
    def _init_hbox_control(self):

        self.hbox_control = QtWidgets.QHBoxLayout()

        self.sl_min_val = 640
        self.sl_max_val = 1280
        self.img_shape = (self.sl_min_val, self.sl_min_val)
        self.sl = ImgSizeSlider(min_val=self.sl_min_val, max_val=self.sl_max_val, step_num=4, orientation='horizontal')

        self.hbox_control.addStretch(1)
        self.hbox_control.addWidget(self.sl)
        self.hbox_control.addWidget(ControlButton("Okay", self.okay_pressed))
        self.hbox_control.addWidget(ControlButton("Update", self.update))
        self.hbox_control.addWidget(ControlButton("Quit", self.quit_default))

    def _init_images(self):
        self.pre_rendered_img_dict = {}

        print("rendering images...", end="")
        for img_size in self.sl.val_list:
            img_size = int(img_size)
            self.pre_rendered_img_dict[img_size] = {}

            (self.pre_rendered_img_dict[img_size]["cropped_data"],
             self.pre_rendered_img_dict[img_size]["full_img"],
             self.pre_rendered_img_dict[img_size]["draw_image"]) = \
                dmk.get_x_from_croped_img(
                    path_img_in=self.img_path,
                    img_shape=(img_size, img_size),
                    window_shape=self.window_shape,
                    step=1.0,
                    color=COLOR_GOOD
                )
        print("ok")

    def __init__(self, config_dict):
        super(WindowClassificationPicture, self).__init__()
        self.setWindowTitle("Plant Disease Recognizer")

        self.img_path = self.choose_picture()
        self.img_name = os.path.splitext(self.img_path)[0]

        self.window_shape = config_dict['windows_shape']
        self.classes = config_dict['classes']

        self._init_hbox_control()
        self._init_images()

        self.main_layout = MyGridLayout(hbox_control=self.hbox_control)
        self.setLayout(self.main_layout)

        self.update_main_layout()
        self.show()

    def clear(self):
        self.main_layout.clear()

    def update_main_layout(self):
        self.clear()

        self.img_shape = (self.sl.value(), self.sl.value())
        (self.cropped_data,
         self.full_img,
         self.draw_image) = \
            (self.pre_rendered_img_dict[self.img_shape[0]]["cropped_data"],
             self.pre_rendered_img_dict[self.img_shape[0]]["full_img"],
             self.pre_rendered_img_dict[self.img_shape[0]]["draw_image"])

        label_list = []
        for x in self.cropped_data['x_data']:
            label_list.append(
                TrainExLabel(
                    x_data=x,
                    classes=self.classes
                )
            )

        self.main_layout.update_grid(
            x_len=int(self.full_img.size[0] / self.cropped_data["x_data"].shape[1]),
            y_len=int(self.full_img.size[1] / self.cropped_data["x_data"].shape[2]),
            label_list=label_list
        )

    def okay_pressed(self):

        y_data = np.empty(0)
        new_classes = {}
        for key, value in self.classes.items():
            new_classes[key] = {'num': 0, 'value': value}

        for label in self.main_layout.label_list:
            y_data = np.append(y_data, self.classes[label.class_name])
            new_classes[label.class_name]['num'] += 1

        y_data.shape = (len(self.cropped_data['x_data']), len(self.classes))

        dmk.json_train_create(
            path="%s.json" % self.img_name,
            cropped_data=self.cropped_data,
            y_data=y_data,
            img_shape=self.full_img.size,
            classes=new_classes
        )

        self.draw_image.save("%s_net.JPG" % self.img_name)

        print("OKAY")

        self.quit_default()
