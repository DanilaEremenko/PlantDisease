from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from pd_lib import data_maker as dmk
import numpy as np
import os
from pd_lib.img_proc import draw_rect_on_image, draw_rect_on_array

COLOR_BAD = 0
COLOR_GOOD = 255


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

    def mouseDoubleClickEvent(self, ev):
        self.change_type()

    def change_type(self):
        self.type = int(not self.type)
        if self.type == 0:
            self.setPixmap(self.good_pixmap)
        elif self.type == 1:
            self.setPixmap(self.bad_pixmap)

        print("type = %d" % self.type)


class ControlButton(QPushButton):
    def __init__(self, text, connect_func):
        super(ControlButton, self).__init__()
        btn_width = 64
        btn_height = 32
        self.resize(btn_width, btn_height)
        self.setText(text)
        self.clicked.connect(connect_func)


class WindowClassificationPicture(QWidget):
    def __init__(self):
        super(WindowClassificationPicture, self).__init__()
        self.setWindowTitle("Plant Disease Recognizer")
        self.img_label = QLabel(self)
        self.setMouseTracking(True)
        picture_path = self.choose_picture()
        self.picture_name = os.path.splitext(picture_path)[0]

        self.x_data, self.x_coord, self.full_img, self.draw_image = dmk.get_x_from_croped_img(
            path_img_in=picture_path,
            img_shape=(768, 768),
            window_shape=(32, 32, 3),
            step=1.0,
            color=COLOR_GOOD
        )
        self.label_list = []
        self.button_init()
        self.show()
        pass

    def button_init(self):

        hbox_control = QtWidgets.QHBoxLayout()
        hbox_control.addStretch(1)
        hbox_control.addWidget(ControlButton("Okay", self.okay_pressed))
        hbox_control.addWidget(ControlButton("Quit", self.quit_pressed))

        x_len = int(self.full_img.size[0] / self.x_data.shape[1])
        y_len = int(self.full_img.size[1] / self.x_data.shape[2])

        hbox_list = []
        i = 0
        for y in range(0, y_len):
            hbox_new = QtWidgets.QHBoxLayout()
            hbox_new.addStretch(1)
            for x in range(0, x_len):
                label_new = TrainExLabel(self.x_data[i])
                hbox_new.addWidget(label_new)
                self.label_list.append(label_new)
                i += 1
            hbox_list.append(hbox_new)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addStretch(1)

        for hbox in hbox_list:
            vbox.addLayout(hbox)
        vbox.addLayout(hbox_control)

        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        self.setLayout(vbox)

        self.setLayout(vbox)
        self.show()

    def choose_picture(self):
        return str(
            QtWidgets.QFileDialog.getOpenFileName(self, "Open *.png, *.jpg file with potato field", None,
                                                  "*.png *.PNG *.jpg *.JPG")[0])

    def okay_pressed(self):

        y_data = np.empty(0)
        i = 0
        for label in self.label_list:
            if label.type == 0:
                y_data = np.append(y_data, [0, 1])
                self.draw_image = draw_rect_on_image(self.draw_image, self.x_coord[i], color=COLOR_GOOD)
            elif label.type == 1:
                y_data = np.append(y_data, [1, 0])
                self.draw_image = draw_rect_on_image(self.draw_image, self.x_coord[i], color=COLOR_BAD)
            i += 1

        y_data.shape = (self.x_data.shape[0], 2)

        dmk.json_create(
            path="%s.json" % self.picture_name,
            x_data=self.x_data,
            y_data=y_data,
            img_shape=self.full_img.size
        )

        self.draw_image.save("%s_net.JPG" % self.picture_name)

        print("OKAY")

        self.quit_pressed()

        pass

    def quit_pressed(self):
        QtCore.QCoreApplication.instance().quit()
        pass
