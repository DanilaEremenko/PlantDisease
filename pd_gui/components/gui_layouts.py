from PyQt5 import QtCore
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QScrollArea, QGroupBox


class MyGridLayout(QVBoxLayout):
    def __init__(self, hbox_control):
        super(MyGridLayout, self).__init__()
        self.hbox_image_list = []
        self.label_list = []

        self.label_layout = QVBoxLayout()
        self.groubBox = QGroupBox()
        self.scroll_area = QScrollArea()
        self.groubBox.setLayout(self.label_layout)
        self.scroll_area.setWidget(self.groubBox)
        self.addWidget(self.scroll_area)
        self.addLayout(self.label_layout)

        self.hbox_control = hbox_control
        self.addLayout(self.hbox_control)

        self.addStretch(1)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)

        self.max_width = 1280
        self.max_height = 960

    def clear(self):
        for hbox in self.hbox_image_list:
            hbox.setParent(None)

        for label in self.label_list:
            label.setParent(None)

        self.hbox_image_list = []
        self.label_list = []

    def update_grid(self, windows_width, window_height, x_len, y_len, label_list):
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
            self.label_layout.addLayout(hbox_line)

        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(min(windows_width, self.max_width))
        self.scroll_area.setFixedHeight(min(window_height, self.max_height))

    def quit_pressed(self):
        QtCore.QCoreApplication.instance().quit()
        pass
