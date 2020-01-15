from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage

from pd_lib.img_proc import draw_rect_on_array


class TrainExLabel(QLabel):
    def __init__(self, x_data, class_num, colors):
        if len(colors) != class_num:
            raise TypeError("len(colors) = %d, class_num = %d" % (len(colors), class_num))
        super(TrainExLabel, self).__init__()
        self.type = 0

        self.x_data = x_data
        self.class_num = class_num
        self.class_pixmaps = []

        for color in colors:
            x_img = draw_rect_on_array(x_data, (1, 1, x_data.shape[0] - 1, x_data.shape[1] - 1), color=color)
            self.class_pixmaps.append(QPixmap.fromImage(QImage(x_img, x_data.shape[0], x_data.shape[1],
                                                               QImage.Format_RGB888)))
        self.setPixmap(self.class_pixmaps[0])
        self.resize(x_data.shape[0], x_data.shape[1])

    def mousePressEvent(self, ev):
        self.change_type()

    def change_type(self):
        self.type += 1
        self.type %= self.class_num
        self.setPixmap(self.class_pixmaps[self.type])
        print("type = %d" % self.type)


class ImageLabel(QLabel):
    def __init__(self, img_arr):
        super(ImageLabel, self).__init__()

        self.pixmap = QPixmap.fromImage(QImage(img_arr, img_arr.shape[0], img_arr.shape[1],
                                               QImage.Format_RGB888))
        self.setPixmap(self.pixmap)
        self.resize(img_arr.shape[0], img_arr.shape[1])
