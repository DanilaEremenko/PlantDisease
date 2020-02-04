from PyQt5.QtWidgets import QLabel, QWidget, QComboBox, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class TrainExLabel(QWidget):
    def __init__(self, x_data, classes):
        super(TrainExLabel, self).__init__()

        self.class_name = list(classes.keys())[0]

        self.x_data = x_data
        self.class_num = len(classes)
        self.img_label = QLabel()

        self.img_label.setPixmap(QPixmap.fromImage(QImage(self.x_data, x_data.shape[0], x_data.shape[1],
                                                          QImage.Format_RGB888)))

        self.resize(x_data.shape[0], x_data.shape[1])

        self.cb = QComboBox()
        self.cb.addItems(list(classes.keys()))
        self.cb.currentIndexChanged.connect(self.change_type)

        layout = QHBoxLayout()
        layout.addWidget(self.img_label, alignment=Qt.AlignTop)
        layout.addWidget(self.cb, alignment=Qt.AlignBottom)

        self.setLayout(layout)

    def mousePressEvent(self, ev):
        self.change_type()

    def change_type(self):
        self.class_name = self.cb.currentText()
        print("type changed to %s" % self.class_name)


class ImageTextLabel(QWidget):
    def __init__(self, img_arr, text):
        super(ImageTextLabel, self).__init__()

        # self.text_label = QLabel()
        # self.text_label.setText(text)

        self.img_label = QLabel()
        self.img_label.setPixmap(
            QPixmap.fromImage(
                QImage(img_arr, img_arr.shape[0], img_arr.shape[1], QImage.Format_RGB888)
            )
        )
        self.img_label.resize(img_arr.shape[0], img_arr.shape[1])

        layout = QHBoxLayout()
        # layout.addWidget(self.text_label)
        layout.addWidget(self.img_label)
        self.setLayout(layout)
