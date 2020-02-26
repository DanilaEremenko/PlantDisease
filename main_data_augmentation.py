"""
Script for train data augmentation via PyQt GUI
"""

import sys
from PyQt5 import QtWidgets
from pd_gui.gui_data_augmentation import WindowMultipleExamples


def main():
    app = QtWidgets.QApplication(sys.argv)

    window_class_pctr = WindowMultipleExamples()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
