"""
Custom PyQt5.QWidget for cropped image visualizing and processing
"""

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QWidget, QLabel

from pd_gui.components.gui_slider import MyScrollArea


class MyGridWidget(QWidget):
    def __init__(self, hbox_control):
        super(MyGridWidget, self).__init__()
        self.hbox_image_list = []
        self.label_list = []

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        ###############################################
        # ---------- left & right init ----------------
        ###############################################
        self.right_layout_width = 200
        window_layouts = QHBoxLayout(self)
        window_layouts.setContentsMargins(0, 0, 0, 0)
        window_layouts.setSpacing(0)
        window_layouts.setAlignment(QtCore.Qt.AlignCenter)

        self.left_layout = QVBoxLayout(self)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(0)
        self.left_layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.right_layout = QVBoxLayout(self)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)
        self.right_layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        window_layouts.addLayout(self.left_layout)
        window_layouts.addLayout(self.right_layout)
        self.layout.addLayout(window_layouts)

        ###############################################
        # --------- elements initializing -------------
        ###############################################
        self.label_layout = QVBoxLayout(self)
        self.label_layout.setContentsMargins(0, 0, 0, 0)
        self.label_layout.setSpacing(0)
        self.label_layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        from screeninfo import get_monitors
        m = get_monitors()[0]
        self.max_width = m.width
        self.max_height = m.height - 100

        self.groubBox = QGroupBox()
        self.left_scroll_area = MyScrollArea()
        self.groubBox.setLayout(self.label_layout)
        self.left_scroll_area.setWidget(self.groubBox)
        self.left_layout.addWidget(self.left_scroll_area)
        self.left_layout.addLayout(self.label_layout)

        self.hbox_control = hbox_control
        self.layout.addLayout(self.hbox_control)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.left_scroll_area.setWidgetResizable(True)
        self.left_scroll_area.setFixedWidth(self.max_width)
        self.left_scroll_area.setFixedHeight(self.max_height)

    def set_offset(self, x, y):
        self.left_scroll_area.verticalScrollBar().setValue(y)
        self.left_scroll_area.horizontalScrollBar().setValue(x)

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
            hbox_line = QHBoxLayout()
            hbox_line.setSpacing(0)
            hbox_line.setContentsMargins(0, 0, 0, 0)
            for x in range(0, x_len):
                hbox_line.addWidget(self.label_list[i])
                i += 1
            self.hbox_image_list.append(hbox_line)
            self.label_layout.addLayout(hbox_line)

        last_hbox_line = QHBoxLayout()
        last_hbox_line.setSpacing(0)
        last_hbox_line.setContentsMargins(0, 0, 0, 0)
        while i < len(label_list):
            last_hbox_line.addWidget(self.label_list[i])
            i += 1
        self.hbox_image_list.append(last_hbox_line)
        self.label_layout.addLayout(last_hbox_line)
        self.update_scroll(windows_width, window_height)

    def update_scroll(self, windows_width, window_height):
        self.left_scroll_area.setWidgetResizable(True)
        self.left_scroll_area.setFixedWidth(min(windows_width, self.max_width) - self.right_layout_width)
        self.left_scroll_area.setFixedHeight(min(window_height, self.max_height))

    def quit_pressed(self):
        QtCore.QCoreApplication.instance().quit()
