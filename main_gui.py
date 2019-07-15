# coding=utf-8
from __future__ import print_function
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import pyqtSlot
from PyQt4.QtGui import QWidget, QLineEdit, QMainWindow
from addition import get_model_from_json, predict_on_dir
import time
import numpy as np


class WindowStart(QWidget):
    btn_size = 100

    def __init__(self, window_choose_dirs):
        super(WindowStart, self).__init__()
        self.window_choose_dirs = window_choose_dirs
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("Plant Disease Recognizer")
        self.setWindowIcon(QtGui.QIcon("res/robot.jpg"))
        self.home()

    def home(self):
        btn_use_existing_NN = QtGui.QPushButton("Use existing NN")
        btn_use_existing_NN.pressed.connect(self.use_existing_NN)
        btn_use_existing_NN.resize(self.btn_size, self.btn_size)

        btn_create_new_NN = QtGui.QPushButton("Create new NN")
        btn_create_new_NN.pressed.connect(self.create_new_NN)
        btn_create_new_NN.resize(self.btn_size, self.btn_size)

        btn_quit = QtGui.QPushButton("Quit")
        btn_quit.clicked.connect(QtCore.QCoreApplication.instance().quit)
        btn_quit.resize(self.btn_size, self.btn_size)

        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(btn_use_existing_NN)
        hbox.addWidget(btn_create_new_NN)
        hbox.addWidget(btn_quit)

        vbox = QtGui.QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)
        self.show()

    @pyqtSlot()
    def use_existing_NN(self):
        self.window_choose_dirs.home(self)
        pass

    @pyqtSlot()
    def create_new_NN(self):
        print("TODO create new NN")
        pass

    @pyqtSlot()
    def load_NN(self):
        print("TODO load NN")
        pass

    @pyqtSlot()
    def choose_data_dir(self):
        print("TODO choose data dir")
        pass

    @pyqtSlot()
    def predict(self):
        print("predict button pressed = ")
        pass


class WindowPreparePredict(QWidget):
    btn_size = 100
    img_shape = (32, 32, 3)

    def __init__(self):
        super(WindowPreparePredict, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("Plant Disease Recognizer")
        self.setWindowIcon(QtGui.QIcon("res/robot.jpg"))
        self.data_dirs = []
        self.plant_name = "potato"
        self.class_marks = np.empty(0)

        btn_add_data_dir = QtGui.QPushButton("Add data dir")
        btn_add_data_dir.pressed.connect(self.add_data_dir)
        btn_add_data_dir.resize(self.btn_size, self.btn_size)

        btn_predict = QtGui.QPushButton("Predict")
        btn_predict.pressed.connect(self.predict)
        btn_predict.resize(self.btn_size, self.btn_size)

        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(btn_add_data_dir)
        hbox.addWidget(btn_predict)

        vbox = QtGui.QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def home(self, back_window):
        self.back_window = back_window
        self.back_window.hide()
        self.show()

        path_to_model = str(QtGui.QFileDialog.getOpenFileNameAndFilter(self, "Open *.json file with saved NN", "",
                                                                       "Json Files (*.json)")[0])

        self.model = get_model_from_json(path_to_model)

    @pyqtSlot()
    def add_data_dir(self):
        print("TODO add data dir")
        data_dir = QtGui.QFileDialog.getExistingDirectory()
        self.data_dirs.append(data_dir)
        pass

    @pyqtSlot()
    def predict(self):
        print("TODO predict")
        if self.data_dirs.__len__() == 0:
            print("No data dirs defined")
            pass

        predict_on_dir(self.model, self.data_dirs, img_shape=self.img_shape)

        self.hide()
        self.back_window.show()
        pass

###########################################################
# ---------------------- someday --------------------------
###########################################################
# class MyTextBox(QWidget):
#     btn_size = 100
#     padding = 100
#
#     def __init__(self, back_window):
#         super(MyTextBox, self).__init__()
#         self.setGeometry(50, 50, 500, 300)
#         self.setWindowTitle("Plant Disease Recognizer")
#         self.setWindowIcon(QtGui.QIcon("res/robot.jpg"))
#         self.data_dirs = []
#         self.textbox = QLineEdit(self)
#         self.textbox.move(self.padding, self.padding)
#         self.textbox.resize(100, 40)
#         self.line = None
#         self.back_window = back_window
#
#         self.back_window.hide()
#         btn_ok = QtGui.QPushButton("Ok")
#         btn_ok.pressed.connect(self.set_val)
#         btn_ok.resize(self.btn_size, self.btn_size)
#
#         hbox = QtGui.QHBoxLayout()
#         hbox.addStretch(1)
#         hbox.addWidget(btn_ok)
#
#         vbox = QtGui.QVBoxLayout()
#         vbox.addStretch(1)
#         vbox.addLayout(hbox)
#
#         self.setLayout(vbox)
#         self.show()
#
#     @pyqtSlot()
#     def set_val(self):
#         self.line = self.textbox.text()
#         self.hide()
#         self.back_window.show()
#
#         pass