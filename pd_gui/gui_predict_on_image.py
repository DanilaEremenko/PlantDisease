from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget

from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_colors import *
from pd_gui.components.gui_slider import ImgSizeSlider
from .components.gui_labels import ImageLabel

from pd_lib.data_maker import get_x_from_croped_img
from pd_lib.addition import get_full_model, predict_and_localize_on_image

import numpy as np
import os.path


class WindowPredictOnImage(QWidget):
    def __init__(self):
        super(WindowPredictOnImage, self).__init__()
        self.window_shape = (32, 32, 3)
        self.sl_min_val = 512
        self.sl_max_val = 2048
        self.setWindowTitle("Plant Disease Recognizer")
        self.elements_init()
        self.choose_picture()
        self.show()
        self.choose_NN()
        pass

    def elements_init(self):
        hbox_control = QtWidgets.QHBoxLayout()
        hbox_control.addStretch(1)
        hbox_control.addWidget(ControlButton("Predict", self.predict))
        hbox_control.addWidget(ControlButton("Update Image", self.update_picture))
        hbox_control.addWidget(ControlButton("Choose model", self.choose_NN))
        hbox_control.addWidget(ControlButton("Choose image", self.choose_picture))
        hbox_control.addWidget(ControlButton("Quit", self.quit_pressed))

        # ------------ adding slider --------------------
        self.hbox_img = QtWidgets.QHBoxLayout()
        self.hbox_img.addStretch(1)

        vbox_slider = QtWidgets.QVBoxLayout()
        self.sl = ImgSizeSlider(min_val=self.sl_min_val, max_val=self.sl_max_val, step_num=4)
        vbox_slider.addWidget(self.sl)

        # ------------- main vbox -------------
        vbox = QtWidgets.QVBoxLayout()
        vbox.addStretch(1)

        vbox.addLayout(vbox_slider)
        vbox.addLayout(self.hbox_img)
        vbox.addLayout(hbox_control)

        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        self.setLayout(vbox)

    def predict(self):
        res_image = predict_and_localize_on_image(model=self.model, cropped_data=self.cropped_data,
                                                  color_1=COLOR_GOOD, color_2=COLOR_BAD,
                                                  image=self.full_img)
        # self.img_label.setParent(None)# TODO add

        # self.img_label = ImageLabel(np.asarray(res_image, dtype='uint8'))

        res_image.show()  # TODO delete

        # self.hbox_img.addWidget(self.img_label)

    def choose_picture(self):
        self.img_path = str(
            QtWidgets.QFileDialog.getOpenFileName(self,
                                                  "Open *.png, *.jpg file with potato field",
                                                  "Datasets/PotatoFields",
                                                  "*.png *.PNG *.jpg *.JPG")[0])

        self.img_shape = (self.sl.value(), self.sl.value())

        self.cropped_data, self.full_img, self.draw_image = get_x_from_croped_img(
            path_img_in=self.img_path,
            img_shape=self.img_shape,
            window_shape=self.window_shape,
            step=1.0,
            color=COLOR_GOOD
        )

    def update_picture(self):
        if not os.path.isfile(self.img_path) or not hasattr(self, "img_path"):
            print("Files with image does't choosed")

        self.img_shape = (self.sl.value(), self.sl.value())

        self.cropped_data, self.full_img, self.draw_image = get_x_from_croped_img(
            path_img_in=self.img_path,
            img_shape=self.img_shape,
            window_shape=(32, 32, 3),
            step=1.0,
            color=COLOR_GOOD
        )

        # if hasattr(self, 'img_label'):# TODO add
        #     self.img_label.setParent(None)

        # self.img_label = ImageLabel(np.asarray(self.full_img, dtype='uint8'))

        self.draw_image.show()  # TODO delete

        # self.hbox_img.addWidget(self.img_label)

    def choose_NN(self):
        self.weights_path = str(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      "Open *.h5 with NN weights",
                                                                      "models",
                                                                      "*.h5 *.H5")[0])
        self.structure_path = str(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                        "Open *.json with NN structure",
                                                                        "models",
                                                                        "*.json *.JSON")[0])

        if os.path.isfile(self.weights_path) and os.path.isfile(self.structure_path):
            self.model = get_full_model(json_path=self.structure_path, h5_path=self.weights_path)
        else:
            print("Files with model weights and model structure does't choosed")

    def quit_pressed(self):
        QtCore.QCoreApplication.instance().quit()
        pass
