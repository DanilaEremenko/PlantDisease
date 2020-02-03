import sys
from PyQt5 import QtWidgets
from pd_gui.gui_train_form import WindowClassificationPicture
import json
import os


def main():
    app = QtWidgets.QApplication(sys.argv)

    if len(sys.argv) != 2:
        raise Exception("Unexpected number of arguments")

    with open(os.path.abspath(sys.argv[1])) as config_fp:
        config_dict = json.load(config_fp)

    window_class_pctr = WindowClassificationPicture(config_dict)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
