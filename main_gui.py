# coding=utf-8
from __future__ import print_function
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import pyqtSlot
from PyQt4.QtGui import QWidget


class Window(QWidget):
    file_NN = None

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("Plant Disease Recognizer")
        self.setWindowIcon(QtGui.QIcon("res/robot.jpg"))
        self.home()

    def home(self):
        btn_size = 100
        btn_choose_NN = QtGui.QPushButton("Choose NN")
        btn_choose_NN.pressed.connect(self.choose_file)
        btn_choose_NN.resize(btn_size, btn_size)

        btn_quit = QtGui.QPushButton("Quit")
        btn_quit.clicked.connect(QtCore.QCoreApplication.instance().quit)
        btn_quit.resize(btn_size, btn_size)

        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(btn_quit)
        hbox.addWidget(btn_choose_NN)

        vbox = QtGui.QVBoxLayout(self)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.show()

        # file_diaglog = QtGui.QFileDialog()

    @pyqtSlot()
    def choose_file(self):
        self.file_NN = QtGui.QFileDialog.getOpenFileName(self, 'Open file')
