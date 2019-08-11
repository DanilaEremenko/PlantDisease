import sys
from PyQt5 import QtWidgets
from gui.gui_dir_predict import WindowPreparePredict, WindowStart
from gui.gui_train_form import WindowClassificationPicture

if __name__ == '__main__':
    #######################################################
    # ------------------ gui ------------------------------
    #######################################################
    app = QtWidgets.QApplication(sys.argv)
    # window_choose_dirs = WindowPreparePredict()
    # window_start = WindowStart(window_choose_dirs)
    window_class_pctr = WindowClassificationPicture()

    sys.exit(app.exec_())
