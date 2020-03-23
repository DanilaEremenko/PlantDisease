"""
Script for train data augmentation via PyQt GUI
"""

import sys
from PyQt5 import QtWidgets
from pd_gui.gui_data_augmentation import WindowMultipleExamples


def main():
    app = QtWidgets.QApplication(sys.argv)

    if len(sys.argv) == 1:
        json_list = None
    else:
        json_list = sys.argv[1:]

    window_class_pctr = WindowMultipleExamples(json_list=json_list)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
