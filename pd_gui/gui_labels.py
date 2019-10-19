from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage

from pd_lib.img_proc import draw_rect_on_array
from .gui_colors import *


class TrainExLabel(QLabel):
    def __init__(self, x_data):
        super(TrainExLabel, self).__init__()
        self.type = 0

        self.x_data = x_data

        x_img_good = draw_rect_on_array(x_data, (1, 1, x_data.shape[0] - 1, x_data.shape[1] - 1), color=COLOR_GOOD)
        self.good_pixmap = QPixmap.fromImage(QImage(x_img_good, x_data.shape[0], x_data.shape[1],
                                                    QImage.Format_RGB888))

        x_img_bad = draw_rect_on_array(x_data, (1, 1, x_data.shape[0] - 1, x_data.shape[1] - 1), color=COLOR_BAD)
        self.bad_pixmap = QPixmap.fromImage(QImage(x_img_bad, x_data.shape[0], x_data.shape[1],
                                                   QImage.Format_RGB888))

        self.setPixmap(self.good_pixmap)
        self.resize(x_data.shape[0], x_data.shape[1])

    def mousePressEvent(self, ev):
        self.change_type()

    def change_type(self):
        self.type = int(not self.type)
        if self.type == 0:
            self.setPixmap(self.good_pixmap)
        elif self.type == 1:
            self.setPixmap(self.bad_pixmap)

        print("type = %d" % self.type)


class ImageLabel(QLabel):
    def __init__(self, img_arr):
        super(ImageLabel, self).__init__()

        self.pixmap = QPixmap.fromImage(QImage(img_arr, img_arr.shape[0], img_arr.shape[1],
                                               QImage.Format_RGB888))
        self.setPixmap(self.pixmap)
        self.resize(img_arr.shape[0], img_arr.shape[1])
