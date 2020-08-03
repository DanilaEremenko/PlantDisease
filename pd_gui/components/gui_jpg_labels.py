"""
Custom PyQt5 GUI labels
"""

from PyQt5.QtWidgets import QLabel, QWidget, QComboBox, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter
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
        img1 = QImage()
        img1.loadFromData(self.x_data, x_data.shape[0], x_data.shape[1], QImage.Format_RGB888)
        pix = QPixmap.fromImage(img1).scaled(label_size[0], label_size[1])
        self.img_label.setPixmap(pix)

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


def getColor(i):
    colors = {
        0: [255, 0, 0],
        1: [255, 255, 0],
        2: [0, 0, 255],
        3: [0, 0, 0]
    }
    return colors[i]


def chooze_filter(decision):
    if decision is None:
        return None
    else:
        procent, type = max(zip(decision, range(len(decision))))
        alpha =256 if type == 3 else int(procent * 128 + 64)
        colors = getColor(type)
        return QColor(colors[0], colors[1], colors[2], alpha)

def paintJpg(image, color_filter):
    if not type == 3:
        p = QPainter(image)
        p.fillRect(image.rect(), color_filter)
        p.end()
    return image


class MergedJPGLabel(QLabel):

    def __init__(self, datas, classes, label_size, decision, coor):
        super(MergedJPGLabel, self).__init__()

        self.classes = classes
        self.class_name = None
        self.sub_class_name = None

        self.background_images = datas.copy()
        self.colored = [False] * len(self.background_images)
        self.color_filter = chooze_filter(decision)
        self.label_size = int(label_size)
        self.zoom = 1
        self.position = coor

    def updateImage(self, size, color_filter=True):
        if color_filter and not self.colored[size]:
            self.colored[size] = True
            return paintJpg(self.background_images[size], self.color_filter)
        else:
            return self.background_images[size]

    # def updateImage(self, size, decision=None,color_filter=True):
    #     if decision is None:
    #         decision = [0, 0, 0, 0]
    #     if not self.background_images[size] is None:
    #         if ((decision[3] == -1) or not self.colored[size]) and color_filter:
    #             self.colored[size] = True
    #             return paintJpg(self.background_images[size], self.color_filter)
    #         else:
    #             return self.background_images[size]
    def get_pos(self):
        return self.position

    def change_type(self, class_name, sub_class_name):
        self.class_name = class_name
        self.sub_class_name = sub_class_name
        print("type changed to '%s:%s'" % (self.class_name, self.sub_class_name))
        self.zoom = 0.9
        self.updateImage(0)

        global last_selected
        last_selected = {'class': class_name, 'sub_class': sub_class_name}
