from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QPixmap, QImage
from pd_lib import data_maker as dmk
from pd_lib.img_proc import draw_rect
import numpy as np
import os


class WindowClassificationPicture(QWidget):
    img_shape = (768, 768)
    btn_size = int(img_shape[0] / 100)
    window_shape = (32, 32, 3)
    step = 1.0
    color_bad = 125
    color_good = 255

    def __init__(self):
        super(WindowClassificationPicture, self).__init__()
        self.setWindowTitle("Plant Disease Recognizer")
        self.img_label = QLabel(self)
        self.home()
        pass

    def home(self):
        self.setMouseTracking(True)

        picture_path = self.choose_picture()
        self.picture_name = os.path.splitext(picture_path)[0]

        x_data, x_coord, full_img, draw_image = dmk.get_x_from_croped_img(
            path_img_in=picture_path,
            img_shape=self.img_shape,
            window_shape=self.window_shape,
            step=self.step,
            color=self.color_good
        )
        self.x_data = x_data
        self.draw_image = draw_image
        self.x_coord = x_coord
        self.colors = np.full(shape=self.x_data.shape[0], fill_value=self.color_good)
        width, height = self.update_image()
        self.resize(width + 1, height + self.btn_size * 4)
        self.img_label.setGeometry(0, 0, width, height)
        self.button_init()
        self.show()
        pass

    def update_image(self):
        im_np = np.asarray(self.draw_image)
        pixmap = QPixmap.fromImage(QImage(im_np, im_np.shape[1], im_np.shape[0],
                                          QImage.Format_RGB888))
        self.img_label.setPixmap(QPixmap(pixmap))
        return pixmap.width(), pixmap.height()

    def button_init(self):
        btn_okay = QtWidgets.QPushButton("Okay")
        btn_okay.pressed.connect(self.okay_pressed)
        btn_okay.resize(self.btn_size, self.btn_size)

        btn_quit = QtWidgets.QPushButton("Quit")
        btn_quit.clicked.connect(self.quit_pressed)
        btn_quit.resize(self.btn_size, self.btn_size)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(btn_okay)
        hbox.addWidget(btn_quit)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        self.setLayout(vbox)

    def choose_picture(self):
        return str(
            QtWidgets.QFileDialog.getOpenFileName(self, "Open *.png, *.jpg file with potato field", None,
                                                  "*.png *.PNG *.jpg *.JPG")[0])

    def mouseDoubleClickEvent(self, QMouseEvent):

        px2 = self.window_shape[0] * (int(QMouseEvent.x() / self.window_shape[0])) + self.window_shape[0]
        py2 = self.window_shape[1] * (int(QMouseEvent.y() / self.window_shape[1])) + self.window_shape[1]
        px1 = px2 - self.window_shape[0]
        py1 = py2 - self.window_shape[1]

        point = (px1, py1, px2, py2)

        color = None
        for i in range(0, self.colors.__len__()):
            if point.__eq__(tuple(self.x_coord[i])):
                print("%s found" % str(point))
                if self.colors[i] == self.color_good:
                    self.colors[i] = self.color_bad
                else:
                    self.colors[i] = self.color_good
                color = self.colors[i]

        if color == None:
            print("%s not found " % str(point))
        else:
            self.draw_image = draw_rect(self.draw_image, point, color=color)
            self.update_image()
        pass

    def okay_pressed(self):
        self.draw_image.save("%s_net.JPG" % self.picture_name)
        y_data = np.empty(0)

        for i in range(0, self.colors.__len__()):
            if self.colors[i] == self.color_good:
                y_data = np.append(y_data, [0, 1])
            else:
                y_data = np.append(y_data, [1, 0])

        y_data.shape = (self.x_data.shape[0], 2)

        dmk.json_create(
            path="%s.json" % self.picture_name,
            x_data=self.x_data,
            y_data=y_data
        )

        print("OKAY")

        self.quit_pressed()

        pass

    def quit_pressed(self):
        QtCore.QCoreApplication.instance().quit()
        pass
