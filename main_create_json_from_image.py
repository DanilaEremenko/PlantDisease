import sys
from PyQt5 import QtWidgets
from pd_gui.gui_train_form import WindowClassificationPicture


def main():
    app = QtWidgets.QApplication(sys.argv)

    window_class_pctr = WindowClassificationPicture()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
