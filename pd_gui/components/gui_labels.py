from pd_lib import img_proc
from PyQt5.QtWidgets import QLabel, QWidget, QComboBox, QHBoxLayout, QMenu
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


##########################################################
# ---------------- train data validation -----------------
##########################################################
class ApartTrainExLabel(QWidget):
    def __init__(self, x_data, classes, label_size):
        super(ApartTrainExLabel, self).__init__()

        self.class_name = list(classes.keys())[0]

        self.x_data = x_data
        self.class_num = len(classes)
        self.img_label = QLabel()
        self.img_label.setPixmap(
            QPixmap
                .fromImage(QImage(self.x_data, x_data.shape[0], x_data.shape[1], QImage.Format_RGB888))
                .scaled(label_size[0], label_size[1])
        )

        self.cb = QComboBox()
        self.cb.addItems(list(classes.keys()))
        self.cb.currentIndexChanged.connect(self.change_type)

        layout = QHBoxLayout()
        layout.addWidget(self.img_label, alignment=Qt.AlignTop)
        layout.addWidget(self.cb, alignment=Qt.AlignBottom)

        self.setLayout(layout)

    def change_type(self):
        self.class_name = self.currentText()
        print("type changed to %s" % self.class_name)


##########################################################
# ---------------- train data initializing ---------------
##########################################################
class MergedTrainExLabel(QLabel):
    def __init__(self, x_data, classes, label_size):
        super(MergedTrainExLabel, self).__init__()

        self.class_names = list(classes.keys())
        self.class_name = None

        self.x_data = x_data
        self.class_num = len(classes)

        self.label_size = label_size
        self.zoom = 1
        self.updateImage(self.label_size)

    def updateImage(self, label_size):
        self.label_size = label_size

        self.setPixmap(
            QPixmap
                .fromImage(
                QImage(
                    img_proc.draw_rect_on_array(
                        img_arr=self.x_data.copy(),
                        points=(1, 1, self.x_data.shape[0] - 1, self.x_data.shape[1] - 1),
                        color=255
                    ),
                    self.x_data.shape[0],
                    self.x_data.shape[1],
                    QImage.Format_RGB888
                )
            ).scaled(*list(map(lambda x: x * self.zoom, self.label_size)))
        )

    def contextMenuEvent(self, event):
        right_click_menu = QMenu(self)
        actions = []

        def addMenuAction(name):
            actions.append(right_click_menu.addAction(name))
            actions[-1].triggered.connect(lambda: self.change_type(name))

        for class_name in self.class_names:
            addMenuAction(class_name)

        right_click_menu.exec_(event.globalPos())

    def change_type(self, class_name):
        self.class_name = class_name
        print("type changed to %s" % self.class_name)
        self.zoom = 0.9
        self.updateImage(label_size=self.label_size)


##########################################################
# ---------------- check NN prediction -------------------
##########################################################
class ImageTextLabel(QWidget):
    def __init__(self, x, text, label_size):
        super(ImageTextLabel, self).__init__()

        self.text_label = QLabel()
        self.text_label.setText(text)

        self.img_label = QLabel()
        self.img_label.setPixmap(
            QPixmap
                .fromImage(QImage(x, x.shape[0], x.shape[1], QImage.Format_RGB888))
                .scaled(label_size[0], label_size[1])
        )
        self.img_label.resize(x.shape[0], x.shape[1])

        layout = QHBoxLayout()
        layout.addWidget(self.img_label)
        layout.addWidget(self.text_label)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 5, 0, 5)
        self.setLayout(layout)
