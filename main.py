import sys
from PyQt4 import QtGui
from main_gui import WindowPreparePredict,WindowStart

if __name__ == '__main__':
    #######################################################
    # ------------------ params ---------------------------
    #######################################################
    wind_width = 400
    wind_height = 500
    app_name = "Plant Disease Recognizer"

    #######################################################
    # ------------------ gui ------------------------------
    #######################################################
    app = QtGui.QApplication(sys.argv)
    window_choose_dirs = WindowPreparePredict()
    window_start = WindowStart(window_choose_dirs)

    sys.exit(app.exec_())
