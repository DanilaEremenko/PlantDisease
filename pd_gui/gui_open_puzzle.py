from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QAction, QApplication
from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_jpg_labels import MergedJPGLabel
from pd_gui.components.gui_layouls_table import MyGridWidget
from pd_gui.gui_window_puzzle import WindowPuzzle
from pd_gui.gui_get_position_photos import GetMozaicMatrix
from pd_lib import data_maker as dmk

import numpy as np
import os

from pd_lib.gui_loading_thread import DownloadOnScreenThread, DownloadListThread, UpdateScreenThread
from pd_lib.image_jpeg_data_maker import read_bin_jpeg
from pd_lib.image_jpeg_data_maker import create_bin_jpeg


class WindowGlobalPuzzle(WindowPuzzle):
    def __init__(self):
        super(WindowGlobalPuzzle, self).__init__()
        self.imgs_row = 0
        self.imgs_line = 0
        self.setWindowTitle("Puzzle Map")
        self.finish_zooming = False
        self.classes = [1, 2, 3]
        self.window_shape = [256, 256, 3]
        self.default_label_size = [256, 256]
        self.img_thumb = [10480, 8192]
        self.imgs_name = []
        self.multiple_size = [20, 150]
        self._init_hbox_control()
        self._init_main_menu()

        self.main_layout = MyGridWidget(hbox_control=self.hbox_control)
        self.setCentralWidget(self.main_layout)
        self.showFullScreen()

        self.last_x = 0
        self.last_y = 0

    def _init_hbox_control(self):
        self.hbox_control = QtWidgets.QHBoxLayout()

        self.zoom_list = [0.0625, 0.125, 0.25, 0.5, 1]
        self.zoom = self.zoom_list[0]
        self.zoom_no = 4

        self.hbox_control.addWidget(ControlButton("Okay", self.okay_pressed, styleSheet='background-color: #0cdb3c'))
        self.hbox_control.addWidget(ControlButton("Crop", self.crop_pressed, styleSheet='background-color: #ebfa78'))
        self.hbox_control.addWidget(ControlButton("Open", self.open_pressed, styleSheet='background-color: #fab978'))
        self.hbox_control.addWidget(ControlButton("Quit", self.quit_default, styleSheet='background-color: #e84a1a'))

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
        self.first_x = event.x()
        self.first_y = event.y()
        # print("event press", event.x(), event.y())
        # print("last ", self.last_x, self.last_y)
        # print("press offset ", self.first_x, self.first_y)

    def mouseMoveEvent(self, event):
        self.v_bar = self.main_layout.scroll_area.verticalScrollBar()
        self.h_bar = self.main_layout.scroll_area.horizontalScrollBar()
        x = self.h_bar.value() + self.first_x - event.x()
        y = self.v_bar.value() + self.first_y - event.y()
        self.last_x = x
        self.last_y = y
        self.first_x = event.x()
        self.first_y = event.y()
        self.main_layout.set_offset(x, y)

    # ------------------------ WHEEL PART -------------------------------------
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            if self.zoom_no < len(self.zoom_list) - 1: self.zoom_no += 1
        else:
            if self.zoom_no > 0: self.zoom_no -= 1
        print('finish_zooming ', self.finish_zooming)
        if self.finish_zooming:
            self.finish_zooming = False
            self.change_zoom(self.zoom_list[self.zoom_no])
            self.move_by_cursor()

    def move_by_cursor(self):
        cursor_x = QtGui.QCursor.pos().x()
        cursor_y = QtGui.QCursor.pos().y()

        window_width = self.main_layout.width()
        window_height = self.main_layout.height()

        rect = list(map(lambda x: x * self.zoom_list[self.zoom_no], self.multiple_size))
        real_image_width = int(rect[0])
        real_image_height = int(rect[1])
        # print("x coor ", cursor_x, window_width, real_image_width)
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

            # print("\n\nZOOM OFFSET:", x, y)
            self.main_layout.set_offset(x, y)

            # TODO maybe someday zoom will work
            # self.last_x = x
            # self.last_y = y

    # ------------------------ ZOOM PART -------------------------------------
    def change_zoom(self, new_zoom):
        self.zoom = new_zoom
        self.screen_updating.zoom_call.emit(new_zoom)

    def _init_images(self):
        QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        # ----------------
        print("rendering images...")
        sum_datas = []
        full_img = [0, 0]
        for path in self.imgs_path:
            for img in path:
                if img != None:
                    x_data_full, full_img = dmk.get_x_from_croped_img(
                        path_to_img=img,
                        window_shape=self.window_shape,
                        step=1.0,
                        verbose=True
                    )
                    sum_datas.append(x_data_full['x_data'].copy())  # x_data_full=picture

        img_line = int(full_img.size[0] / self.window_shape[0])
        img_row = int(full_img.size[1] / self.window_shape[1])
        scap = int(img_line * img_row * self.count_photos)
        self.connected_x_data_full = np.empty([scap, *self.window_shape[0:-1], 3], dtype='uint8')

        # self.multiple_size[0] = self.imgs_line * img_line
        # self.multiple_size[1] = self.imgs_row * img_row
        # print(scap, img_line, img_row, self.multiple_size)
        element_to = -1
        empty_frame = 0
        # начало

        puzzle_mask = np.empty((self.imgs_row * img_row, self.imgs_line * img_line), dtype=np.bool)
        print('puzzle_mask ', puzzle_mask.shape)
        # TODO remove duplicate cycles
        for i in range(self.imgs_row):
            for j in range(self.imgs_line):
                for n in range(img_row):
                    for m in range(img_line):
                        curr = False if self.imgs_path[i][j] is None else True
                        puzzle_mask[n + i * img_row][m + j * img_line] = curr

        for row in range(0, self.imgs_row):
            for frames_row in range(0, img_row):
                for line in range(0, self.imgs_line):
                    for frames_line in range(0, img_line):
                        photo_from = line + row * self.imgs_line - int(empty_frame / (img_line * img_row))
                        element_from = img_line * frames_row + frames_line
                        x = int(element_from % 20)
                        y = int(element_from / 20)
                        if puzzle_mask[y + row * 15][x + line * 20]:
                            element_to += 1
                            self.connected_x_data_full[element_to] = sum_datas[photo_from][element_from]
                        else:
                            empty_frame += 1
                        # print(scap, frames_line, frames_row, line, row, '-----', element_to)

        print()
        np.save('output/mask_photos', puzzle_mask)
        np.save('output/bin_photos', self.connected_x_data_full)
        self.jpgs_name = create_bin_jpeg(self.connected_x_data_full.copy())
        QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        print("ok")

    def choose_and_render_image(self):
        dir = self.choose_file_dir()
        if not dir == '':
            gmm = GetMozaicMatrix()
            self.imgs_path, self.count_photos = gmm.get_matrix(dir)
            if not self.count_photos == 0:
                self.clear()
                self.img_width, self.img_height = gmm.get_resolution()
                for line in self.imgs_path:
                    for img in line:
                        if img != None:
                            self.imgs_name.append(os.path.splitext(img))

                self.imgs_line = len(self.imgs_path[0])
                self.imgs_row = len(self.imgs_path)
                print('photos ', self.imgs_line, self.imgs_row, len(self.imgs_name))
                self._init_images()
            else:
                print('No foto files')
                self.choose_and_render_image()

    def clear(self):
        self.main_layout.clear()

    def crop_pressed(self):
        self.choose_and_render_image()

    def finish_zoom(self, state):
        self.finish_zooming = state

    def open_pressed(self):
        self.clear()
        file = self.choose_file_jpgs_massive()
        if not file is None:
            self.jpgs = read_bin_jpeg(file)
            list_loading = DownloadListThread(self.jpgs, self.main_layout)
            self.screen_updating = UpdateScreenThread(self.main_layout, self.zoom)
            self.screen_updating.zoom_call.connect(self.screen_updating.zooming)
            self.screen_updating.zoom_end.connect(self.finish_zoom)
            list_loading.signal.connect(self.screen_updating.displayS)
            list_loading.start()

    def okay_pressed(self):
        self.main_layout.table.repaint()
        # y_data = np.empty(0)
        # for class_name in self.classes.keys():
        #     for sub_class_name in self.classes[class_name]:
        #         self.classes[class_name][sub_class_name]['num'] = 0
        #
        # # Now we're going to store only examples of diseased plants
        # x_data_full = {
        #     'x_data': np.empty(0, dtype='uint8'),
        #     'x_coord': [],
        #     'longitudes': [],
        #     'latitudes': []
        # }
        #
        # ex_num = 0
        # for x, label in zip(self.x_data_full['x_data'], self.main_layout.label_list):
        #     if label.class_name is not None:
        #         x_data_full['x_data'] = np.append(x_data_full['x_data'], x)
        #         y_data = np.append(y_data, self.classes[label.class_name][label.sub_class_name]['value'])
        #         self.classes[label.class_name][label.sub_class_name]['num'] += 1
        #         ex_num += 1
        #
        # x_data_full['x_data'].shape = (ex_num, *self.window_shape)
        # y_data.shape = (len(x_data_full['x_data']), 1)

        print("OKAY")

        # self.choose_and_render_image()
