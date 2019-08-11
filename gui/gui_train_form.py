from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QLineEdit, QMainWindow
from data_maker import get_x_from_croped_img


class WindowClassificationPicture(QWidget):
    btn_size = 100
    img_shape = (512, 512)
    window_shape = (32, 32, 3)
    step = 1.0

    def __init__(self):
        super(WindowClassificationPicture, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("Plant Disease Recognizer")
        self.home()

    def home(self):
        self.show()
        self.choose_picture()
        x_data, x_coord, full_img, draw_image = get_x_from_croped_img(
            path_img_in=self.picture_path,
            img_shape=self.img_shape,
            window_shape=self.window_shape,
            step=self.step
        )
        print("x_data.shape = %s" % str(x_data.shape))

        QtCore.QCoreApplication.instance().quit()

    @pyqtSlot()
    def choose_picture(self):
        self.picture_path = str(
            QtWidgets.QFileDialog.getOpenFileName(self, "Open *.png, *.jpg file with potato field", "",
                                                  "Picure (*.png, *.jpg)")[0])
