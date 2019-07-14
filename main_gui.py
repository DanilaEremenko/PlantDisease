# coding=utf-8
from __future__ import print_function
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import pyqtSlot
from PyQt4.QtGui import QWidget, QLineEdit, QMainWindow
from addition import get_model_from_json


class Window_Start(QWidget):
    btn_size = 100

    def __init__(self, window_choose_dirs):
        super(Window_Start, self).__init__()
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
        self.hide()
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


class Window_Choose_Dirs(QWidget):
    btn_size = 100

    def __init__(self):
        super(Window_Choose_Dirs, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("Plant Disease Recognizer")
        self.setWindowIcon(QtGui.QIcon("res/robot.jpg"))
        self.data_dirs = []

    def home(self, back_window):
        self.back_window = back_window
        # path_to_model = str(QtGui.QFileDialog.getOpenFileNameAndFilter(self, "Open *.json file with saved NN", "",
        #                                                               "Json Files (*.json)")[0])
        self.plant_name = MyTextBox().read_value()
        print("plant name = " % self.plant_name)
        # print("loading model from %s" % path_to_model)
        # self.model = get_model_from_json(path_to_model)

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
        self.show()

    @pyqtSlot()
    def add_data_dir(self):
        print("TODO add data dir")
        data_dir = QtGui.QFileDialog.getExistingDirectory()
        self.data_dirs.append(data_dir)
        pass

    @pyqtSlot()
    def predict(self):
        print("TODO predict")
        data = {}
        for data_dir in self.data_dirs:
            data['']
        self.hide()
        self.back_window.show()
        pass


class MyTextBox(QWidget):
    btn_size = 100

    def __init__(self):
        super(MyTextBox, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("Plant Disease Recognizer")
        self.setWindowIcon(QtGui.QIcon("res/robot.jpg"))
        self.data_dirs = []
        self.textbox = QLineEdit(self)
        self.textbox.move(self.padding, self.padding)
        self.textbox.resize(100, 40)
        self.line = ""

    def read_value(self, back_window):
        btn_ok = QtGui.QPushButton("Ok")
        btn_ok.pressed.connect(self.set_val)
        btn_ok.resize(self.btn_size, self.btn_size)

        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(btn_ok)

        vbox = QtGui.QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)
        self.show()

    @pyqtSlot()
    def set_val(self):
        self.line = self.textbox.text()
        pass
