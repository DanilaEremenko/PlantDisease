import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from pd_gui.components.gui_jpg_labels import MergedJPGLabel


class UpdateScreenThread(QThread):
    zoom_call = pyqtSignal(object)
    zoom_end = pyqtSignal(bool)

    def __init__(self, main_layout, zoom):
        super(UpdateScreenThread, self).__init__()
        self.label_list = []
        self.line_lenght = 0
        self.main_layout = main_layout
        self.zoom = zoom
        self.label_size = [0, 0]
        self.output = []

    def zooming(self, zoom):
        self.zoom = zoom
        self.run()

    def displayS(self, label_list, ll, def_size, output):
        self.label_list = label_list
        self.line_lenght = ll
        self.label_size = def_size
        self.output = output
        self.zoom_end.emit(True)

    def run(self):
        self.main_layout.resizeTable(edge=self.label_size[0] * self.zoom)
        print('am in ', self.label_size[0] * self.zoom)
        x = 0
        for label in self.label_list:
            updated = label.updateImage(label_size=[self.label_size[0] * self.zoom, self.label_size[1] * self.zoom],
                                        decision=self.output[x])

            self.main_layout.update_cell(x=x, y=x * self.line_lenght, image=updated)
            x += 1
            self.msleep(1)
        self.zoom_end.emit(True)


class DownloadListThread(QThread):
    signal = pyqtSignal(object, object, object, object)

    def __init__(self, data, main_layout):
        super(DownloadListThread, self).__init__()
        self.jpgs = data
        self.classes = [1, 2, 3]
        self.default_label_size = [256, 256]
        self.label_list = []
        self.main_layout = main_layout
        self.output = np.load('output/bin_u_photos.npy')
        self.mask = np.load('output/mask_photos.npy')
        self.multiple_size = [0, 0]
        self.multiple_size[0] = self.mask.shape[1]
        self.multiple_size[1] = self.mask.shape[0]
        self.main_layout.resizeTable(size=self.multiple_size, edge=self.default_label_size[0])

    def run(self):
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                self.label_list.append(MergedJPGLabel(
                    x_data=self.jpgs[j + i * self.mask.shape[1]] if self.mask[i, j] else None,
                    classes=self.classes,
                    label_size=self.default_label_size))
                screen_loading = DownloadOnScreenThread(self.label_list[-1],
                                                        self.main_layout,
                                                        j,
                                                        i,
                                                        self.output[j + i * self.mask.shape[1]])
                screen_loading.start()
        self.signal.emit(self.label_list, self.mask.shape[0], self.default_label_size, self.output)


class DownloadOnScreenThread(QThread):
    exchange_Signal = pyqtSignal()
    start_paint_Signal = pyqtSignal()

    def __init__(self, data, main_layout, x, y, decision):
        super(DownloadOnScreenThread, self).__init__()
        self.label = data
        self.main_layout = main_layout
        self.default_label_size = [256, 256]
        self.x = x
        self.y = y
        self.decision = decision

    def run(self):
        updated = self.label.updateImage(label_size=self.default_label_size, decision=self.decision)
        self.main_layout.update_cell(x=self.x, y=self.y, image=updated)
