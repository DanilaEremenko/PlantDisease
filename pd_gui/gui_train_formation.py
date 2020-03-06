"""
PyQt GUI for main_create_json_from_image.py
"""

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QAction, QApplication
from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_labels import MergedTrainExLabel
from pd_gui.components.gui_layouts import MyGridWidget
from pd_gui.gui_window_interface import WindowInterface

from pd_lib import data_maker as dmk

import numpy as np
import os


class WindowClassificationPicture(WindowInterface):
    def _init_hbox_control(self):

        self.hbox_control = QtWidgets.QHBoxLayout()

        self.zoom_list = [0.25, 0.5, 0.75, 1]
        self.zoom = self.zoom_list[0]
        self.zoom_no = 0

        self.hbox_control.addWidget(
            ControlButton("Okay", self.okay_pressed, styleSheet='background-color: #0cdb3c'))
        self.hbox_control.addWidget(
            ControlButton("Choose image", self.choose_and_render_image, styleSheet='background-color: #ffbe25'))
        self.hbox_control.addWidget(
            ControlButton("Quit", self.quit_default, styleSheet='background-color: #e84a1a'))

    def _init_main_menu(self):

        mainMenu = self.menuBar()
        zoomMenu = mainMenu.addMenu('Zoom')

        def add_zoom_to_menu(new_zoom):
            newAct = QAction('Zoom %d %%' % (zoom * 100), self)
            newAct.triggered.connect(lambda: self.change_zoom(new_zoom))
            zoomMenu.addAction(newAct)

        for zoom in self.zoom_list:
            add_zoom_to_menu(zoom)

    # ------------------------ MOUSE DRAGGING PART -------------------------------------
    def mousePressEvent(self, event):
        # TODO to fix
        rect = list(map(lambda x: x * self.zoom_list[self.zoom_no], self.full_img.size))
        self.first_x = max(0, min(int(rect[0]), event.x() + self.last_x))
        self.first_y = max(0, min(int(rect[1]), event.y() + self.last_y))
        print("event ", event.x(), event.y())
        print("last ", self.last_x, self.last_y)
        print("press offset ", self.first_x, self.first_y)

    def mouseMoveEvent(self, event):
        self.v_bar = self.main_layout.scroll_area.verticalScrollBar()
        self.h_bar = self.main_layout.scroll_area.horizontalScrollBar()
        rect = list(map(lambda x: x * self.zoom_list[self.zoom_no], self.full_img.size))
        if self.main_layout.width() < rect[0]:
            x = self.first_x - event.x()
            y = self.first_y - event.y()
            self.last_x = x
            self.last_y = y
            print("\n\nDRAG OFFSET:", x, y)
            self.main_layout.set_offset(x, y)

    # ------------------------ WHEEL PART -------------------------------------
    #   mouse wheel event scrollо
    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                if self.zoom_no < len(self.zoom_list) - 1: self.zoom_no += 1
            else:
                if self.zoom_no > 0: self.zoom_no -= 1
            self.change_zoom(self.zoom_list[self.zoom_no])
            self.move_by_cursor()

    def move_by_cursor(self):
        cursor_x = QtGui.QCursor.pos().x()
        cursor_y = QtGui.QCursor.pos().y()

        window_width = self.main_layout.width()
        window_height = self.main_layout.height()

        rect = list(map(lambda x: x * self.zoom_list[self.zoom_no], self.full_img.size))
        real_image_width = int(rect[0])
        real_image_height = int(rect[1])
        print("x coor ", cursor_x, window_width, real_image_width)
        if (real_image_width < window_width | real_image_height < window_height):
            print("nothing to move")
        else:
            koef_x = (cursor_x) / real_image_width
            koef_y = (cursor_y) / real_image_height

            offset_x = (real_image_width - window_width) * koef_x
            offset_y = (real_image_height - window_height) * koef_y

            # TODO famous math constant 4 and 2
            x = int(offset_x * 4)
            y = int(offset_y * 2)

            print("\n\nZOOM OFFSET:", x, y)
            self.main_layout.set_offset(x, y)

            self.last_x = x
            self.last_y = y

    # ------------------------ ZOOM PART -------------------------------------
    def change_zoom(self, new_zoom):
        self.zoom = new_zoom
        print('new zoom = %d' % self.zoom)
        self.update_main_layout()

    def _init_images(self):
        self.pre_rendered_img_dict = {}
        QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))

        print("rendering image...")
        self.x_data_full, self.full_img = \
            dmk.get_x_from_croped_img(
                path_img_in=self.img_path,
                window_shape=self.window_shape,
                step=1.0,
                color=255,
                verbose=True
            )
        QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        print("ok")

    def _init_label_list(self):
        self.label_list = []
        for x in self.x_data_full['x_data']:
            self.label_list.append(
                MergedTrainExLabel(
                    x_data=x,
                    classes=self.classes,
                    label_size=self.default_label_size
                )
            )

    def __init__(self):
        super(WindowClassificationPicture, self).__init__()
        self.setWindowTitle("Plant Disease Recognizer")

        with open(self.choose_json(content_title='config data')) as config_fp:
            config_dict = json.load(config_fp)

        # self.img_path = self.choose_picture()
        # self.img_name = os.path.splitext(self.img_path)[0]

        self.window_shape = config_dict['window_shape']
        self.classes = config_dict['classes']
        self.default_label_size = config_dict['qt_label_size']
        self.img_thumb = config_dict['img_thumb']

        self._init_hbox_control()
        self._init_main_menu()

        self.main_layout = MyGridWidget(hbox_control=self.hbox_control)
        self.setCentralWidget(self.main_layout)
        self.showFullScreen()

        self.choose_and_render_image()

        # for offset calculation
        self.last_x = 0
        self.last_y = 0

    def choose_and_render_image(self):
        self.clear()

        self.img_path = self.choose_picture()

        if self.img_path != '':
            self.img_name = os.path.splitext(self.img_path)[0]
            self._init_images()
            self._init_label_list()
            self.update_main_layout()

    def clear(self):
        self.main_layout.clear()

    def update_main_layout(self):
        self.clear()

        for label in self.label_list:
            label.updateImage(label_size=list(map(lambda x: x * self.zoom, self.default_label_size)))

        self.main_layout.update_grid(
            windows_width=self.main_layout.max_width,
            window_height=self.main_layout.max_height,
            x_len=int(self.full_img.size[0] / self.x_data_full["x_data"].shape[1]),
            y_len=int(self.full_img.size[1] / self.x_data_full["x_data"].shape[2]),
            label_list=self.label_list
        )

        print("img_size = %s\nex_num = %d\n" % (str(self.full_img.size), len(self.x_data_full['x_data'])))

    def okay_pressed(self):

        y_data = np.empty(0)
        for class_name in self.classes.keys():
            for sub_class_name in self.classes[class_name]:
                self.classes[class_name][sub_class_name]['num'] = 0

        # Now we're going to store only examples of diseased plants
        x_data_full = {
            'x_data': np.empty(0, dtype='uint8'),
            'x_coord': [],
            'longitudes': [],
            'latitudes': []
        }

        ex_num = 0
        for x, label in zip(self.x_data_full['x_data'], self.main_layout.label_list):
            if label.class_name is not None:
                x_data_full['x_data'] = np.append(x_data_full['x_data'], x)
                y_data = np.append(y_data, self.classes[label.class_name][label.sub_class_name]['value'])
                self.classes[label.class_name][label.sub_class_name]['num'] += 1
                ex_num += 1

        x_data_full['x_data'].shape = (ex_num, *self.window_shape)
        y_data.shape = (len(x_data_full['x_data']), 1)

        dmk.json_train_create(
            path="%s.json" % self.img_name,
            x_data_full=x_data_full,
            y_data=y_data,
            img_shape=self.full_img.size,
            classes=self.classes
        )

        print("OKAY")

        self.choose_and_render_image()
