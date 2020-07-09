import os

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from pd_gui.components.gui_jpg_labels import MergedJPGLabel
from pd_gui.gui_get_position_photos import GetMozaicMatrix
from pd_lib.image_jpeg_data_maker import create_bin_jpeg
from pd_lib import data_maker as dmk


class UpdateScreenThread(QThread):
    zoom_call = pyqtSignal(object)
    zoom_end = pyqtSignal(bool)

    def __init__(self, main_layout, zoom_arr):
        super(UpdateScreenThread, self).__init__()
        self.label_list = []
        self.line_lenght = 0
        self.main_layout = main_layout
        self.zoom_arr = zoom_arr
        self.label_size = 0
        self.output = []

    def zooming(self, zoom):
        self.zoom = zoom

    def displayS(self, label_list, ll, def_size, output):
        self.label_list = label_list
        self.line_lenght = ll
        print('finish load')
        self.label_size = def_size
        self.output = output
        self.zoom_end.emit(True)

    def run(self):
        self.main_layout.resizeTable(edge=int(self.label_size * self.zoom_arr[self.zoom]))
        print('am in ', int(self.label_size * self.zoom_arr[self.zoom]), self.zoom_arr[self.zoom], self.zoom)
        x = 0
        self.zoom_end.emit(False)
        for label in self.label_list:
            updated = label.updateImage(size=self.zoom, decision=self.output[x])
            self.main_layout.update_cell(x=x % self.line_lenght,
                                         y=int(x / self.line_lenght),
                                         image=updated)
            x += 1
        self.zoom_end.emit(True)


class DownloadListThread(QThread):
    signal = pyqtSignal(object, object, object, object)
    progress_signal = pyqtSignal(int)

    def __init__(self, datas, main_layout, zoom_array):
        super(DownloadListThread, self).__init__()
        self.zoom_jpgs = datas
        self.classes = [1, 2, 3]
        self.zooms = zoom_array
        self.default_label_size = int(256 * self.zooms[0])
        self.label_list = []
        self.main_layout = main_layout
        self.output = np.load('output/bin_u_photos.npy')
        self.mask = np.load('output/mask_photos.npy')
        self.multiple_size = [0, 0]
        self.multiple_size[0] = self.mask.shape[1]
        self.multiple_size[1] = self.mask.shape[0]
        self.main_layout.resizeTable(size=self.multiple_size, edge=self.default_label_size)

    def sort_jpgs(self, j, i):
        ara = []
        for m in range(len(self.zooms)):
            img = QImage()
            img.loadFromData(self.zoom_jpgs[m][j + i * self.mask.shape[1]] if self.mask[i, j] else None)
            ara.append(img)
        return ara

    def run(self):
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                self.label_list.append(MergedJPGLabel(
                    datas=self.sort_jpgs(j, i),
                    classes=self.classes,
                    label_size=self.default_label_size))
                updated = self.label_list[-1].updateImage(size=self.zooms.index(1),
                                                          decision=self.output[j + i * self.mask.shape[1]])
                self.main_layout.update_cell(x=j, y=i, image=updated)
                self.usleep(10)
            self.progress_signal.emit(i * 100 / self.mask.shape[0])
        self.signal.emit(self.label_list, self.mask.shape[1], self.default_label_size, self.output)


class CroplerThread(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self, imgs_path, count_photos, window_shape, zoom_list, imgs_line, imgs_row):
        super(CroplerThread, self).__init__()
        self.imgs_path = imgs_path
        self.count_photos = count_photos
        self.window_shape = window_shape
        self.zoom_list = zoom_list
        self.imgs_line = imgs_line
        self.imgs_row = imgs_row

    # self.progress_signal.emit(i * 100 / self.mask.shape[0])

    def run(self):
        print('start thread')
        sum_datas = []
        full_img = [0, 0]
        imgs_row = self.imgs_row
        imgs_line = self.imgs_line
        for path in self.imgs_path:
            for img in path:
                if img != None:
                    x_data_full, full_img = dmk.get_x_from_croped_img(
                        path_to_img=img,
                        window_shape=self.window_shape,
                        step=1.0,
                        verbose=True
                    )
                    sum_datas.append(x_data_full['x_data'].copy())
        img_line = int(full_img.size[0] / self.window_shape[0])
        img_row = int(full_img.size[1] / self.window_shape[1])
        scap = int(img_line * img_row * self.count_photos)
        connected_x_data_full = np.empty([scap, *self.window_shape[0:-1], 3], dtype='uint8')

        element_to = -1
        empty_frame = 0

        puzzle_mask = np.empty((imgs_row * img_row, imgs_line * img_line), dtype=np.bool)
        print('puzzle_mask ', puzzle_mask.shape)
        # TODO remove duplicate cycles
        for i in range(imgs_row):
            for j in range(imgs_line):
                for n in range(img_row):
                    for m in range(img_line):
                        curr = False if self.imgs_path[i][j] is None else True
                        puzzle_mask[n + i * img_row][m + j * img_line] = curr

        for row in range(0, imgs_row):
            for frames_row in range(0, img_row):
                for line in range(0, imgs_line):
                    for frames_line in range(0, img_line):
                        photo_from = line + row * imgs_line - int(empty_frame / (img_line * img_row))
                        element_from = img_line * frames_row + frames_line
                        x = int(element_from % 20)
                        y = int(element_from / 20)
                        if puzzle_mask[y + row * 15][x + line * 20]:
                            element_to += 1
                            connected_x_data_full[element_to] = sum_datas[photo_from][element_from]
                        else:
                            empty_frame += 1
        print('masses ', len(sum_datas), len(sum_datas[0]), len(connected_x_data_full))
        np.save('output/mask_photos', puzzle_mask)
        np.save('output/bin_photos', connected_x_data_full)
        create_bin_jpeg(self, connected_x_data_full.copy(), self.zoom_list)
