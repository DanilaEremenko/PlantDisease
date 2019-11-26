from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget

from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_colors import *
from .components.gui_labels import ImageLabel

from pd_lib.data_maker import get_x_from_croped_img
from pd_lib.addition import get_full_model, predict_and_localize_on_image

import numpy as np
import os.path


class WindowPredictOnImage(QWidget):
    def __init__(self):
        super(WindowPredictOnImage, self).__init__()
        self.setWindowTitle("Plant Disease Recognizer")
        self.button_init()
        self.choose_picture()
        self.show()
        self.choose_NN()
        pass

    def button_init(self):
        hbox_control = QtWidgets.QHBoxLayout()
        hbox_control.addStretch(1)
        hbox_control.addWidget(ControlButton("Predict", self.predict))
        hbox_control.addWidget(ControlButton("Choose model", self.choose_NN))
        hbox_control.addWidget(ControlButton("Choose image", self.choose_picture))
        hbox_control.addWidget(ControlButton("Quit", self.quit_pressed))

        self.hbox_img = QtWidgets.QHBoxLayout()
        self.hbox_img.addStretch(1)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addStretch(1)

        vbox.addLayout(self.hbox_img)
        vbox.addLayout(hbox_control)

        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        self.setLayout(vbox)

    def predict(self):
        res_image = predict_and_localize_on_image(model=self.model, x_data=self.x_data, x_coord=self.x_coord,
                                                  color_1=COLOR_GOOD, color_2=COLOR_BAD,
                                                  image=self.full_img)
        self.img_label.setParent(None)

        self.img_label = ImageLabel(np.asarray(res_image, dtype='uint8'))

        res_image.show()  # TODO delete

        self.hbox_img.addWidget(self.img_label)

    def choose_picture(self):
        img_path = str(
            QtWidgets.QFileDialog.getOpenFileName(self,
                                                  "Open *.png, *.jpg file with potato field",
                                                  "Datasets/PotatoFields",
                                                  "*.png *.PNG *.jpg *.JPG")[0])

        if os.path.isfile(img_path):
            self.x_data, self.x_coord, self.full_img, self.draw_image = get_x_from_croped_img(
                path_img_in=img_path,
                img_shape=(768, 768),
                window_shape=(32, 32, 3),
                step=1.0,
                color=COLOR_GOOD
            )

            if hasattr(self, 'img_label'):
                self.img_label.setParent(None)

            self.img_label = ImageLabel(np.asarray(self.full_img, dtype='uint8'))

            self.full_img.show()  # TODO delete

            self.hbox_img.addWidget(self.img_label)
        else:
            print("Files with image does't choosed")

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
