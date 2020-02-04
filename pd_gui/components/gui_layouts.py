from PyQt5 import QtCore
from PyQt5.QtWidgets import QScrollBar, QVBoxLayout, QHBoxLayout


class MyGridLayout(QVBoxLayout):
    def __init__(self, hbox_control):
        super(MyGridLayout, self).__init__()

        self.hbox_image_list = []
        self.label_list = []

        self.hbox_control = hbox_control

        self.addStretch(1)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)

        # TODO works incorrect
        self.scroll_right_box = QVBoxLayout()
        self.scroll_bar = QScrollBar()
        self.scroll_bar.setMaximum(255)
        self.scroll_right_box.addWidget(self.scroll_bar)

        pass

    def clear(self):
        for hbox in self.hbox_image_list:
            hbox.setParent(None)
        for label in self.label_list:
            label.setParent(None)
        self.hbox_control.setParent(None)
        self.hbox_image_list = []
        self.label_list = []

    def update_grid(self, x_len, y_len, label_list):
        self.clear()
        self.label_list = label_list
        # -------------------- init image --------------------------
        i = 0
        for y in range(0, y_len):
            hbox_new = QHBoxLayout()
            hbox_new.addStretch(1)
            for x in range(0, x_len):
                hbox_new.addWidget(self.label_list[i])
                i += 1
            self.hbox_image_list.append(hbox_new)

        # -------------------- add boxes --------------------------
        for hbox_line in self.hbox_image_list:
            self.addLayout(hbox_line)
        self.addLayout(self.hbox_control)
        print("image updated")

    def quit_pressed(self):
        QtCore.QCoreApplication.instance().quit()
        pass
