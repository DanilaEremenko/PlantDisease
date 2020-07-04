import json
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow


class WindowPuzzle(QMainWindow):
    def choose_file_jpgs_massive(self):
        return str(QtWidgets.QFileDialog.getOpenFileName(self, "Open *.bin file ", None, "*.bin *.BIN")[0])

    def choose_file_dir(self):
        return str(QtWidgets.QFileDialog().getExistingDirectory(self, "Open *.png, *.jpg file's with potato field"))

    def _init_hbox_control(self):
        raise Exception("isn't implemented")

    def quit_default(self):
        QtCore.QCoreApplication.instance().quit()

    def __init__(self):
        super(WindowPuzzle, self).__init__()
        self._init_hbox_control()
