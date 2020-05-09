"""
PyQt GUI for main_full_system_test.py
"""

import json
import os

from PyQt5 import QtWidgets
from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_layouts import MyGridWidget

from pd_lib.keras_addition_ import get_full_model
from pd_lib import data_maker as dmk

from .gui_window_interface import WindowInterface
from pd_gui.components.gui_labels import ImageTextLabel

import numpy as np


class WindowPredictOnImage(WindowInterface):
    def _init_hbox_control(self):
        self.hbox_control = QtWidgets.QHBoxLayout()
        self.hbox_control.addStretch(1)
        self.hbox_control.addWidget(ControlButton("Update Image", self.update_main_layout))
        self.hbox_control.addWidget(ControlButton("Choose model", self.choose_NN))
        self.hbox_control.addWidget(ControlButton("Choose image", self._parse_image))
        self.hbox_control.addWidget(ControlButton("Quit", self.quit_default))

    def _parse_image(self):
        self.picture_path = self.choose_picture()
        x_cropped, full_img = dmk.get_x_from_croped_img(
            path_to_img=self.picture_path,
            window_shape=(256, 256, 3),
            img_thumb=self.img_thumb

        )
        self.x_data = x_cropped['x_data']

    def _define_max_key_len(self):
        self.max_key_len = 0
        for key, value in self.classes.items():
            if len(key) > self.max_key_len:
                self.max_key_len = len(key)

    def __init__(self):
        super(WindowPredictOnImage, self).__init__()
        with open(os.path.abspath('config_full_system.json')) as config_fp:
            config_dict = json.load(config_fp)
            self.classes = config_dict['classifier']['classes']
            self.img_thumb = config_dict['gui']['img_thumb']
            self.label_size = config_dict['gui']['qt_label_size']
            self.model = get_full_model(**config_dict['classifier']['args'])

        self._define_max_key_len()

        self._parse_image()

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

        def get_key_by_answer(pos_code):
            answer = {'mae': 9999, 'key': None, 'value': 0}
            for key in self.classes.keys():
                mae = np.average(abs((self.classes[key]['value'] - pos_code)))
                if mae < answer['mae']:
                    answer['mae'] = mae
                    answer['key'] = key
                    answer['value'] = max(pos_code)
            return answer

        def add_spaces(word, new_size):  # TODO fix gui label alignment
            while len(word) < new_size:
                word += '_'
            return word

        label_list = []
        for x, y_answer in zip(self.x_data, self.model.predict(self.x_data)):
            answer = get_key_by_answer(pos_code=y_answer)
            answer['key'] = add_spaces(answer['key'], new_size=self.max_key_len)

            label_list.append(
                ImageTextLabel(
                    x=x,
                    text='%s %.2f' % (answer['key'], answer['value']),
                    label_size=self.label_size
                )
            )
        rect_len = int(np.sqrt(len(self.x_data)))
        self.main_layout.update_grid(
            windows_width=self.frameGeometry().width(),
            window_height=self.frameGeometry().height(),
            x_len=rect_len,
            y_len=rect_len,
            label_list=label_list
        )

    def choose_NN(self):
        self.weights_path = str(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      "Open *.h5 with NN weights",
                                                                      "models",
                                                                      "*.h5 *.H5")[0])
        self.structure_path = str(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                        "Open *.json with NN structure",
                                                                        "models",
                                                                        "*.json *.JSON")[0])

        if os.path.isfile(self.weights_path) and os.path.isfile(self.structure_path):
            self.model = get_full_model(json_path=self.structure_path, h5_path=self.weights_path)
        else:
            print("Files with model weights and model structure does't choosed")
