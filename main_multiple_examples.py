import sys
from PyQt5 import QtWidgets
from pd_gui.gui_multiple_image import WindowMultipleExamples

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window_class_pctr = WindowMultipleExamples()

    sys.exit(app.exec_())
