import json

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget


class WindowInterface(QWidget):
    def choose_picture(self):
        return str(
            QtWidgets.QFileDialog.getOpenFileName(self, "Open *.png, *.jpg file with potato field", None,
                                                  "*.png *.PNG *.jpg *.JPG")[0])

    def choose_json(self, content_title):
        return str(
            QtWidgets.QFileDialog.getOpenFileName(self, "Open *.json file with %s" % content_title, None,
                                                  "*.json *.JSON")[0])

    def load_dict_from_json_with_keys(self, key_list):
        is_valid = False
        while not is_valid:
            with open(self.choose_json(content_title='config data')) as config_fp:
                config_dict = json.load(config_fp)
                is_valid = True
                for key in key_list:
                    if key not in config_dict.keys():
                        is_valid = False
                        break

        return config_dict

    def _init_hbox_control(self):
        raise Exception("isn't implemented")

    def quit_default(self):
        QtCore.QCoreApplication.instance().quit()

    def __init__(self):
        super(WindowInterface, self).__init__()
        self._init_hbox_control()
