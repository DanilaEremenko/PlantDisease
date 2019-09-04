import sys
from PyQt5 import QtWidgets
from pd_gui.gui_train_form import WindowClassificationPicture

if __name__ == '__main__':
    #######################################################
    # ------------------ pd_gui ------------------------------
    #######################################################
    app = QtWidgets.QApplication(sys.argv)

    window_class_pctr = WindowClassificationPicture()

    sys.exit(app.exec_())

    # x_train, y_train = dmk.json_load("Datasets/PotatoFields/plan_train/DJI_0246.json")
    #
    # print("x_train.shape = %s" % str(x_train.shape))
    # print("y_train.shape = %s" % str(y_train.shape))
