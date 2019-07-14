import sys
from PyQt4 import QtGui
from main_gui import *

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
    window_choose_dirs = Window_Choose_Dirs()
    window_start = Window_Start(window_choose_dirs)

    sys.exit(app.exec_())
