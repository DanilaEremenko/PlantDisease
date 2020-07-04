"""
Script for open window to observe a photo puzzle
"""

import sys
from PyQt5 import QtWidgets
from pd_gui.gui_open_puzzle import WindowGlobalPuzzle


def main():
    app = QtWidgets.QApplication(sys.argv)
    WindowGlobalPuzzle()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
