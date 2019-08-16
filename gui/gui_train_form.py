from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QLineEdit, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage
from data_maker import get_x_from_croped_img
import numpy as np


class WindowClassificationPicture(QWidget):
    btn_size = 100
    img_shape = (768, 768)
    window_shape = (32, 32, 3)
    step = 1.0
    color_bad = 125
    color_good = 255

    def __init__(self):
        super(WindowClassificationPicture, self).__init__()
        self.setWindowTitle("Plant Disease Recognizer")
        self.img_label = QLabel(self)
        self.home()

    def home(self):
        self.setMouseTracking(True)
        self.choose_picture()
        x_data, x_coord, full_img, draw_image = get_x_from_croped_img(
            path_img_in=self.picture_path,
            img_shape=self.img_shape,
            window_shape=self.window_shape,
            step=self.step,
            color=self.color_good
        )
        self.full_data = zip(x_coord, zip(x_data, np.full(shape=x_data.shape[0], fill_value=self.color_good)))
        for coord, data in self.full_data:
            print("coord = %s, color = %d" % (str(coord), int(data[1])))
        im_np = np.asarray(draw_image)
        pixmap = QPixmap.fromImage(QImage(im_np, im_np.shape[1], im_np.shape[0],
                                          QImage.Format_RGB888))
        self.img_label.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width() + 1, pixmap.height())
        self.img_label.setGeometry(0, 0, pixmap.width(), pixmap.height())
        self.show()

    @pyqtSlot()
    def choose_picture(self):
        self.picture_path = str(
            QtWidgets.QFileDialog.getOpenFileName(self, "Open *.png, *.jpg file with potato field", None,
                                                  "*.png *.PNG *.jpg *.JPG")[0])

    def mouseDoubleClickEvent(self, QMouseEvent):
        x = self.window_shape[0] * (int(QMouseEvent.x() / self.window_shape[0])) + self.window_shape[0]
        y = self.window_shape[1] * (int(QMouseEvent.y() / self.window_shape[1])) + self.window_shape[1]

        print('Mouse coords: ( %d : %d )' % (x, y))
