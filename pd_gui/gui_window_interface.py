from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget


class WindowInterface(QWidget):
    def choose_picture(self):
        return str(
            QtWidgets.QFileDialog.getOpenFileName(self, "Open *.png, *.jpg file with potato field", None,
                                                  "*.png *.PNG *.jpg *.JPG")[0])

    def _init_hbox_control(self):
        raise Exception("isn't implemented")

    def quit_default(self):
        QtCore.QCoreApplication.instance().quit()

    def __init__(self):
        super(WindowInterface, self).__init__()
        self._init_hbox_control()
