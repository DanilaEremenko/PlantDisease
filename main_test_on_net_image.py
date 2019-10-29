import sys
from PyQt5 import QtWidgets
from pd_gui.gui_predict_on_image import WindowPredictOnImage

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window_class_pctr = WindowPredictOnImage()

    sys.exit(app.exec_())
