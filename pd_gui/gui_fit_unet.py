"""
PyQt GUI for visualizing predictions of NN in main_fit_CNN.py
"""

from PyQt5 import QtWidgets
from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_layouts import MyGridWidget

from .gui_window_interface import WindowInterface
from pd_gui.components.gui_labels import ImageTextLabel


class WindowShowUnetFitting(WindowInterface):
    def _init_hbox_control(self):
        self.hbox_control = QtWidgets.QHBoxLayout()
        self.hbox_control.addStretch(1)
        self.hbox_control.addWidget(ControlButton("Update", self.update_main_layout))
        self.hbox_control.addWidget(ControlButton("Quit", self.quit_default))

    def __init__(self, x_data, y_data, y_predicted):
        super(WindowShowUnetFitting, self).__init__()

        config_dict = self.load_dict_from_json_with_keys(key_list=['qt_label_size'])
        self.label_size = config_dict['qt_label_size']

        self.main_layout = MyGridWidget(hbox_control=self.hbox_control)
        self.setCentralWidget(self.main_layout)
        self.showFullScreen()
        self.update_main_layout(x_data, y_data, y_predicted)

    def clear(self):
        self.main_layout.clear()

    def update_main_layout(self, x_data, y_data, y_predicted):
        self.clear()

        label_list = []
        for x, y, y_answer in zip(x_data, y_data, y_predicted):
            label_list.append(
                ImageTextLabel(
                    x=x,
                    text='blop',
                    label_size=self.label_size
                )
            )
            label_list.append(
                ImageTextLabel(
                    x=y,
                    text='blop',
                    label_size=self.label_size
                )
            )
            label_list.append(
                ImageTextLabel(
                    x=y_answer,
                    text='blop',
                    label_size=self.label_size
                )
            )

        x_len = 3
        self.main_layout.update_grid(
            windows_width=self.frameGeometry().width(),
            window_height=self.frameGeometry().height(),
            x_len=x_len,
            y_len=int(x_data.shape[0] * 3 / x_len),
            label_list=label_list
        )
