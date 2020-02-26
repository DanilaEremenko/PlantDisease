"""
Custom PyQt5 GUI labels
"""

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
last_selected = None


class MergedTrainExLabel(QLabel):
    def __init__(self, x_data, classes, label_size):
        super(MergedTrainExLabel, self).__init__()

        self.classes = classes
        self.class_name = None
        self.sub_class_name = None

        self.x_data = x_data

        self.label_size = label_size
        self.zoom = 1
        self.updateImage(self.label_size)

    def updateImage(self, label_size):
        self.label_size = label_size

        img_arr = self.x_data.copy()
        for i in range(1, 3):
            img_arr = img_proc.draw_rect_on_array(
                img_arr=img_arr,
                points=(i, i, self.x_data.shape[0] - i, self.x_data.shape[1] - i),
                color=255
            )

        self.setPixmap(
            QPixmap
                .fromImage(
                QImage(
                    img_arr,
                    img_arr.shape[0],
                    img_arr.shape[1],
                    QImage.Format_RGB888
                )
            ).scaled(*list(map(lambda x: x * self.zoom, self.label_size)))
        )

    def contextMenuEvent(self, event):
        right_click_menu = QMenu(self)
        if last_selected is not None:
            action = right_click_menu.addAction(last_selected['sub_class'])
            right_click_menu.addSeparator()
            action.triggered.connect(lambda: self.change_type(last_selected['class'], last_selected['sub_class']))
        added_menu = []

        def addMenuAction(class_name, sub_class_name):
            for rec_name in cur_rec_list:
                if rec_name not in added_menu:
                    right_click_ptr[0] = right_click_ptr[0].addMenu(rec_name)
                    added_menu.append(rec_name)

            action = right_click_ptr[0].addAction(sub_class_name)
            action.triggered.connect(lambda: self.change_type(class_name, sub_class_name))

        def findValue(cur_sub_dict, key):
            if 'value' not in cur_sub_dict:
                right_click_ptr[0] = right_click_menu
                cur_rec_list.append(key)
                for next_key in cur_sub_dict.keys():
                    findValue(cur_sub_dict[next_key], next_key)
            else:
                addMenuAction(cur_rec_list[0], key)

        for top_class_key in self.classes.keys():
            right_click_ptr = [right_click_menu]  # TODO so bad move
            cur_rec_list = []
            findValue(self.classes[top_class_key], top_class_key)

        right_click_menu.exec_(event.globalPos())

    def change_type(self, class_name, sub_class_name):
        self.class_name = class_name
        self.sub_class_name = sub_class_name
        print("type changed to '%s:%s'" % (self.class_name, self.sub_class_name))
        self.zoom = 0.9
        self.updateImage(label_size=self.label_size)

        global last_selected
        last_selected = {'class': class_name, 'sub_class': sub_class_name}


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
