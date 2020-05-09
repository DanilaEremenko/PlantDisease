"""
PyQt GUI for visualizing predictions of NN in main_fit_CNN.py
"""

from PyQt5 import QtWidgets
from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_layouts import MyGridWidget

from .gui_window_interface import WindowInterface
from pd_gui.components.gui_labels import ImageTextLabel

import numpy as np


class WindowShowPredictions(WindowInterface):
    def _init_hbox_control(self):
        self.hbox_control = QtWidgets.QHBoxLayout()
        self.hbox_control.addStretch(1)
        self.hbox_control.addWidget(ControlButton("Update", self.update_main_layout))
        self.hbox_control.addWidget(ControlButton("Quit", self.quit_default))

    def _define_max_key_len(self):
        self.max_key_len = 0
        for key, value in self.classes.items():
            if len(key) > self.max_key_len:
                self.max_key_len = len(key)

    def __init__(self, x_data, y_data, y_predicted, classes):
        super(WindowShowPredictions, self).__init__()

        config_dict = self.load_dict_from_json_with_keys(key_list=['qt_label_size'])
        self.label_size = config_dict['qt_label_size']

        self.x_data = x_data
        self.y_data = y_data
        self.y_predicted = y_predicted
        self.classes = classes

        self._define_max_key_len()

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
        for x, y, y_answer in zip(self.x_data, self.y_data, self.y_predicted):
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
