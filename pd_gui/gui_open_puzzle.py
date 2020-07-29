import subprocess

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QAction, QProgressBar
from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_layouls_table import MyGridWidget
from pd_gui.gui_window_puzzle import WindowPuzzle
from pd_gui.gui_get_position_photos import GetMozaicMatrix
from pd_lib.loading_threads import DownloadListThread, UpdateScreenThread, SlicerThread
from pd_lib.image_jpeg_data_maker import read_bin_jpeg
import os

class WindowGlobalPuzzle(WindowPuzzle):
    def __init__(self):
        super(WindowGlobalPuzzle, self).__init__()
        if not os.path.exists('output'):
            os.mkdir('output')
        self.setWindowTitle("Puzzle Map")
        self.finish_zooming = False
        self.classes = [1, 2, 3]
        self.window_shape = [256, 256, 3]
        self.default_label_size = [256, 256]
        self.img_thumb = [10480, 8192]

        self.multiple_size = [20, 150]
        self._init_hbox_control()
        self._init_main_menu()

        self.main_layout = MyGridWidget(hbox_control=self.hbox_control, progress=self.progress)
        self.setCentralWidget(self.main_layout)
        self.showFullScreen()

        self.last_x = 0
        self.last_y = 0

        # self._init_hbox_control()

    def _init_hbox_control(self):

        self.hbox_control = QtWidgets.QHBoxLayout()
        self.hbox_progress = QtWidgets.QHBoxLayout()
        self.zoom_list = [1, 0.5, 0.25, 0.125, 0.0625]
        # self.zoom_list = [0.0625, 0.125, 0.25, 0.5, 1]
        self.zoom_no = 0
        self.zoom = self.zoom_list[self.zoom_no]
        self.progress = QProgressBar()
        # self.progress.setGeometry(50, 50, 500, 300)
        self.hbox_progress.addWidget(self.progress)
        self.hbox_control.addWidget(ControlButton("Slice", self.crop_pressed, styleSheet='background-color: #ebfa78'))
        self.hbox_control.addWidget(ControlButton("Open photo", self.open_pressed, styleSheet='background-color: #0cdb3c'))
        self.hbox_control.addWidget(ControlButton("Open with filter", self.open_f_pressed, styleSheet='background-color: #fab978'))
        self.hbox_control.addWidget(ControlButton("Learn by photos", self.start_learning, styleSheet='background-color: #fa56dd'))
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
        print('finish_zooming ', self.finish_zooming)
        if self.finish_zooming:
            if event.angleDelta().y() < 0:
                if self.zoom_no < len(self.zoom_list) - 1: self.zoom_no += 1
            else:
                if self.zoom_no > 0: self.zoom_no -= 1
            print('set zoom ', self.zoom_no)
            self.change_zoom(self.zoom_no)
            self.move_by_cursor()

    def move_by_cursor(self):
        cursor_x = QtGui.QCursor.pos().x()
        cursor_y = QtGui.QCursor.pos().y()

        window_width = self.main_layout.width()
        window_height = self.main_layout.height()

        rect = list(map(lambda x: x * self.zoom_list[self.zoom_no], self.multiple_size))
        real_image_width = int(rect[0])
        real_image_height = int(rect[1])
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
        self.screen_updating.start()

    def choose_and_render_image(self):

        dir = self.choose_file_dir()
        if not dir == '':
            imgs_name = []
            gmm = GetMozaicMatrix()
            imgs_path, count_photos = gmm.get_matrix(dir)
            if not count_photos == 0:
                self.clear()
                img_width, img_height = gmm.get_resolution()
                n=0
                for line in imgs_path:
                    for img in line:
                        if img != None:
                            imgs_name.append(os.path.splitext(img))
                            self.update_progress(100*n/len(line))
                            n +=1
                imgs_line = len(imgs_path[0])
                imgs_row = len(imgs_path)

                self.crop = SlicerThread(imgs_path, count_photos, self.window_shape, self.zoom_list, imgs_line, imgs_row)
                self.crop.progress_signal.connect(self.update_progress)
                self.crop.start()
                # self._init_images(imgs_path, count_photos)
            else:
                self.choose_and_render_image()

    def clear(self):
        self.main_layout.clear()

    def crop_pressed(self):
        self.choose_and_render_image()

    def update_progress(self, percent):
        self.progress.setValue(1 + percent)

    def finish_zoom(self, state):
        self.finish_zooming = state
        print('finish load ', state)

    def start_learning(self):
        subprocess.call('main_full_system_predict.py -i output/bin_photos.npy -o output/bin_u_photos.npy -n 3000', shell=True)

    def open_pressed(self):
        self.start_loading(False)

    def open_f_pressed(self):
        self.start_loading(True)

    def start_loading(self, color_filter):
        # self.clear()
        self.jpgs_names = []
        for i in self.zoom_list:
            self.jpgs_names.append("output\jpeg_array_" + str(i) + ".bin")
        if not self.jpgs_names is None:
            self.jpgs_mass = read_bin_jpeg(self.jpgs_names)
            self.list_loading = DownloadListThread(self.jpgs_mass, self.main_layout, self.zoom_list, color_filter)
            self.screen_updating = UpdateScreenThread(self.main_layout, self.zoom_list)
            self.screen_updating.zoom_call.connect(self.screen_updating.zooming)
            self.screen_updating.zoom_end.connect(self.finish_zoom)
            self.list_loading.progress_signal.connect(self.update_progress)
            self.list_loading.signal.connect(self.screen_updating.displayS)
            self.list_loading.start()


    def okay_pressed(self):
        self.main_layout.resizeTable(edge=16)
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
