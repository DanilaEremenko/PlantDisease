import sys
from PyQt5 import QtWidgets
from gui.gui_train_form import WindowClassificationPicture
import data_maker as dmk

if __name__ == '__main__':
    #######################################################
    # ------------------ gui ------------------------------
    #######################################################
    app = QtWidgets.QApplication(sys.argv)

    window_class_pctr = WindowClassificationPicture()

    sys.exit(app.exec_())

    # x_train, y_train = dmk.json_load("Datasets/PotatoFields/plan_train/DJI_0246.json")
    #
    # print("x_train.shape = %s" % str(x_train.shape))
    # print("y_train.shape = %s" % str(y_train.shape))
