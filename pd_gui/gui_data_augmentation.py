"""
PyQt GUI for main_data_augmentation.py
"""

from PyQt5 import QtWidgets
from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_layouts import MyGridWidget

import pd_lib.data_maker as dmk
from .gui_window_interface import WindowInterface
from pd_gui.components.gui_labels import ImageTextLabel

import json
import os
import numpy as np


class WindowMultipleExamples(WindowInterface):
    def _init_hbox_control(self):
        self.hbox_control = QtWidgets.QHBoxLayout()
        self.hbox_control.addStretch(1)
        self.hbox_control.addWidget(ControlButton("Okay", self.okay_pressed))
        self.hbox_control.addWidget(ControlButton("Update", self.update_main_layout))
        self.hbox_control.addWidget(ControlButton("Multiple", self.multiple_pressed))
        self.hbox_control.addWidget(ControlButton("Quit", self.quit_default))

    def _init_data_from_json(self, json_for_multiple):
        with open(json_for_multiple) as train_json_fp:
            train_json = dict(json.load(train_json_fp))

            self.classes = {}
            for class_name in train_json['classes'].keys():
                for sub_class_name in train_json['classes'][class_name]:
                    self.classes[sub_class_name] = train_json['classes'][class_name][sub_class_name]

            self.x_data = np.array(train_json["x_data"], dtype='uint8')
            self.y_data = np.array(train_json["y_data"])
            self.longitudes, self.latitudes = train_json["longitudes"], train_json["latitudes"]
            self.img_shape = train_json["img_shape"]

    def _define_max_class(self):
        self.max_class = {'name': None, 'num': 0, 'value': None}
        self.max_key_len = 0
        for key, value in self.classes.items():
            if self.classes[key]['num'] > self.max_class['num']:
                self.max_class['name'] = key
                self.max_class['num'] = self.classes[key]['num']
                self.max_class['value'] = self.classes[key]['value']
            if len(key) > self.max_key_len:
                self.max_key_len = len(key)

    def __init__(self):
        super(WindowMultipleExamples, self).__init__()

        with open(self.choose_json(content_title='config gui data')) as gui_config_fp:
            self.label_size = json.load(gui_config_fp)['qt_label_size']

        with open(self.choose_json(content_title='config augmentation data')) as aug_config_fp:
            alghs_dict = json.load(aug_config_fp)['algorithms']
            self.arg_dict = {
                'use_noise': alghs_dict['noise']['use'],
                'intensity_noise_list': alghs_dict['noise']['val_list'],
                'use_deform': alghs_dict['deform']['use'],
                'k_deform_list': alghs_dict['deform']['val_list'],
                'use_blur': alghs_dict['blur']['use'],
                'rad_list': alghs_dict['blur']['val_list']
            }

        json_for_multiple = self.choose_json(content_title='train_data')
        self.json_name = os.path.splitext(json_for_multiple)[0]

        self._init_data_from_json(json_for_multiple)
        self._define_max_class()

        self.main_layout = MyGridWidget(hbox_control=self.hbox_control)
        self.setCentralWidget(self.main_layout)

        self.showFullScreen()
        self.update_main_layout()

    def clear(self):
        self.main_layout.clear()

    def update_main_layout(self):
        self.clear()

        def get_key_by_value(value):
            for key in self.classes.keys():
                if (self.classes[key]['value'] == value).all():
                    return key
            raise Exception('No value == %s' % str(value))

        def add_spaces(word, new_size):  # TODO fix gui label alignment
            while len(word) < new_size:
                word += '_'
            return word

        label_list = []
        for x, y in zip(self.x_data, self.y_data):
            label_list.append(
                ImageTextLabel(
                    x=x,
                    text=add_spaces(get_key_by_value(value=y), new_size=self.max_key_len),
                    label_size=self.label_size
                )
            )
        rect_len = int(np.sqrt(len(self.x_data)))
        self.main_layout.update_grid(
            windows_width=self.main_layout.max_width,
            window_height=self.main_layout.max_height,
            x_len=rect_len,
            y_len=rect_len,
            label_list=label_list
        )

    def okay_pressed(self):
        out_json_path = "%s_multiple.json" % self.json_name
        print("Save to %s" % out_json_path)
        dmk.json_train_create(
            path=out_json_path,
            x_data_full={"x_data": self.x_data, "longitudes": self.longitudes, "latitudes": self.latitudes},
            y_data=self.y_data,
            img_shape=None,
            classes=self.classes
        )

        self.quit_default()

    def multiple_pressed(self):
        for key, value in self.classes.items():
            if self.classes[key]['num'] < self.max_class['num']:
                old_class_size = len(self.x_data)
                self.x_data, self.y_data = dmk.multiple_class_examples(x_train=self.x_data, y_train=self.y_data,
                                                                       class_for_multiple=self.classes[key]['value'],
                                                                       **self.arg_dict,
                                                                       max_class_num=self.max_class['num'])

                new_ex_num = len(self.x_data) - old_class_size
                print('%s : generated %d new examples' % (key, new_ex_num))
                self.classes[key]['num'] = 0
                for y in self.y_data:
                    if ((y.__eq__(self.classes[key]['value'])).all()):
                        self.classes[key]['num'] += 1

            else:
                print('%s : generated %d new examples (class_size == max_size)' % (key, 0))
        print("---------------------------------")
