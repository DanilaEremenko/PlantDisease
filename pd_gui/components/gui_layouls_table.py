from PyQt5 import QtCore
from PyQt5.QtGui import QBrush
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QTableWidget, QTableWidgetItem

from pd_gui.components.gui_slider import MyScrollArea


class MyGridWidget(QWidget):
    def __init__(self, hbox_control, progress):
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
        self.max_height = m.height - 75
        self.scroll_area = MyScrollArea()
        self.table = MyTable()
        self.scroll_area.setWidget(self.table)

        self.left_layout.addWidget(self.scroll_area)
        self.left_layout.addLayout(self.label_layout)

        self.layout.addWidget(progress, alignment=QtCore.Qt.AlignCenter)

        self.hbox_control = hbox_control
        self.layout.addLayout(self.hbox_control)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(self.max_width)
        self.scroll_area.setFixedHeight(self.max_height)

    def updateTable(self, Column, Row):
        self.table.setColumnCount(Column)
        self.table.setRowCount(Row)
        self.table.setShowGrid(False)
        v = self.table.verticalHeader()
        h = self.table.horizontalHeader()
        v.hide()
        h.hide()

    def resizeTable(self, edge, size=None):
        if not size == None:
            self.updateTable(size[0], size[1])
            print('new size set', size)

        self.table.verticalHeader().setMinimumSectionSize(edge)
        self.table.horizontalHeader().setMinimumSectionSize(edge)
        self.table.verticalHeader().setMaximumSectionSize(edge)
        self.table.horizontalHeader().setMaximumSectionSize(edge)
        print('update edge table ', edge)

    def set_offset(self, x, y):
        self.scroll_area.verticalScrollBar().setValue(y)
        self.scroll_area.horizontalScrollBar().setValue(x)

    def clear(self):
        for hbox in self.hbox_image_list:
            hbox.setParent(None)

        for label in self.label_list:
            label.setParent(None)

        self.hbox_image_list = []
        self.label_list = []

    def update_cell(self, x, y, image):
        # -------------------- init image --------------------------
        thumbnail = QTableWidgetItem()
        thumbnail.setBackground(QBrush(image))
        self.table.setItem(y, x, thumbnail)
        self.table.viewport().update()
        # self.update_scroll(windows_width, window_height)

    def update_scroll(self, windows_width, window_height):
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(min(windows_width, self.max_width))
        self.scroll_area.setFixedHeight(min(window_height, self.max_height))

    def quit_pressed(self):
        QtCore.QCoreApplication.instance().quit()


class MyTable(QTableWidget):
    def __init__(self):
        super().__init__()

    def wheelEvent(self, ev):
        if ev.type() == QtCore.QEvent.Wheel:
            ev.ignore()
