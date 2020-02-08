import sys
from PyQt5 import QtWidgets
from pd_gui.gui_window_predict_on_image import WindowPredictOnImage


def main():
    app = QtWidgets.QApplication(sys.argv)

    window_class_pctr = WindowPredictOnImage()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
