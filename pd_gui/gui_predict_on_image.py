"""
PyQt GUI for main_full_system_test.py
"""

import json
import os
import time

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QProgressBar, QLabel

from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_layouts import MyGridWidget

from pd_lib.keras_addition_ import get_full_model
from pd_main_part.classifiers import get_classifier_by_name
from pd_main_part.clusterers import get_clusterer_by_name
from pd_main_part.preprocessors import get_preprocessor_by_name

from .gui_window_interface import WindowInterface
from pd_gui.components.gui_labels import ImageTextLabel

import numpy as np


class WindowPredictOnImage(WindowInterface):
    ##############################################################
    # ---------------- init stuff --------------------------------
    ##############################################################
    def __init__(self):
        super(WindowPredictOnImage, self).__init__()
        with open(os.path.abspath('config_full_system.json')) as config_fp:
            self.config_dict = json.load(config_fp)

            # load layers
            self.clusterer = get_clusterer_by_name(
                self.config_dict['clusterer']['name'],
                self.config_dict['clusterer']['args']
            )
            if self.config_dict['preprocessor']['use']:
                self.segmentator = get_preprocessor_by_name(
                    self.config_dict['preprocessor']['name'],
                    self.config_dict['preprocessor']['args'])
            self.classifier = get_classifier_by_name(
                self.config_dict['classifier']['name'],
                self.config_dict['classifier']['args'])
            self.bad_key = self.config_dict['classifier']['bad_key']

            self._define_max_key_len()
            self._parse_image()

            self.main_layout = MyGridWidget(hbox_control=self.hbox_control)
            self.pbar = QProgressBar(self)
            self.main_layout.layout.addWidget(self.pbar)
            self.right_text_labels = []

            self.setCentralWidget(self.main_layout)
            self.showFullScreen()
            self.update_main_layout()

    def _init_hbox_control(self):
        self.hbox_control = QtWidgets.QHBoxLayout()
        self.hbox_control.addStretch(1)
        self.hbox_control.addWidget(ControlButton("Predict", self.update_main_layout))
        self.hbox_control.addWidget(ControlButton("Choose model", self.choose_NN))
        self.hbox_control.addWidget(ControlButton("Choose image", self._parse_image))
        self.hbox_control.addWidget(ControlButton("Quit", self.quit_default))

    def _parse_image(self):
        self.picture_path = self.choose_picture()
        x_cropped = self.clusterer.cluster(self.picture_path)
        self.x_data = x_cropped['x_data']

    def _define_max_key_len(self):
        self.max_key_len = 0
        for key, value in self.classifier.classes.items():
            if len(key) > self.max_key_len:
                self.max_key_len = len(key)

    def clear(self):
        self.main_layout.clear()
        for text_label in self.right_text_labels:
            text_label.setParent(None)

    def update_main_layout(self):
        self.clear()

        self.predict_thread = PredictThread(self)
        fake_timer = FakeTimer(self)
        fake_timer.valueChanged.connect(self.predict_thread.valueChanged.emit)

        self.predict_thread.valueChanged.connect(self.pbar.setValue)
        self.predict_thread.canDraw.connect(self.draw_result)
        self.predict_thread.fakeTimerToStop.connect(fake_timer.terminate)

        self.main_layout.update_scroll(
            windows_width=self.frameGeometry().width(),
            window_height=self.frameGeometry().height(),
        )

        # self.x_data = self.x_data[:4]
        self.predict_thread.start()
        fake_timer.start()

    def draw_result(self):
        def get_key_by_answer(pos_code, bad_key):
            answer = {'mae': 9999, 'key': bad_key, 'value': 0}
            if sum(pos_code) > 0:
                for key in self.classifier.classes.keys():
                    mae = np.average(abs((self.classifier.classes[key]['value'] - pos_code)))
                    if mae < answer['mae']:
                        answer['mae'] = mae
                        answer['key'] = key
                        answer['value'] = max(pos_code)
            return answer

        def add_spaces(word, new_size):  # TODO fix gui label alignment
            while len(word) < new_size:
                word += '_'
            return word

        label_list = []

        classes_dict = self.classifier.classes.copy()
        classes_dict[self.bad_key] = {}
        for key in [*classes_dict.keys(), self.bad_key]:
            classes_dict[key]['num'] = 0
            classes_dict[key]['indexes'] = []

        for x, y_answer in zip(self.x_data, self.predict_thread.y_answer):
            answer = get_key_by_answer(pos_code=y_answer, bad_key=self.bad_key)

            classes_dict[answer['key']]['num'] += 1

            # classes_dict[answer['key']]['indexes'].append(i)
            # answer['key'] = "%d: %s" % (i, answer['key'])

            answer['key'] = add_spaces(answer['key'], new_size=self.max_key_len)

            label_list.append(
                ImageTextLabel(
                    x=x,
                    text='%s %.2f' % (answer['key'], answer['value']),
                    label_size=self.config_dict['gui']['qt_label_size']
                )
            )

        self.right_text_labels = []
        for key in classes_dict:
            text_label = QLabel()
            text_label.setText("%s: %d" % (key, classes_dict[key]['num']))
            text_label.setAlignment(QtCore.Qt.AlignLeft)
            self.right_text_labels.append(text_label)
            self.main_layout.right_layout.addWidget(text_label)

        rect_len = int(np.sqrt(len(self.x_data)))
        self.main_layout.update_grid(
            windows_width=self.frameGeometry().width(),
            window_height=self.frameGeometry().height(),
            x_len=rect_len,
            y_len=rect_len,
            label_list=label_list
        )

    def choose_NN(self):
        self.weights_path = str(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                      "Open *.h5 with NN weights",
                                                                      "models",
                                                                      "*.h5 *.H5")[0])
        self.structure_path = str(QtWidgets.QFileDialog.getOpenFileName(self,
                                                                        "Open *.json with NN structure",
                                                                        "models",
                                                                        "*.json *.JSON")[0])

        if os.path.isfile(self.weights_path) and os.path.isfile(self.structure_path):
            self.classifier = get_full_model(json_path=self.structure_path, h5_path=self.weights_path)
        else:
            print("Files with model weights and model structure does't choosed")


class PredictThread(QThread):
    canDraw = QtCore.pyqtSignal()
    valueChanged = QtCore.pyqtSignal(int)
    fakeTimerToStop = QtCore.pyqtSignal()
    taskFinished = QtCore.pyqtSignal()

    def __init__(self, parent):
        QThread.__init__(self, parent)
        self.mw = parent

    def run(self):
        start_time = time.time()

        import keras.backend.tensorflow_backend as tb
        tb._SYMBOLIC_SCOPE.value = True

        self.y_answer = self.mw.classifier.predict(self.mw.x_data)
        self.canDraw.emit()
        self.fakeTimerToStop.emit()
        self.valueChanged.emit(100)
        self.taskFinished.emit()
        print('full_time  = %.2f' % (time.time() - start_time))


class FakeTimer(QThread):
    valueChanged = QtCore.pyqtSignal(int)
    taskFinished = QtCore.pyqtSignal()

    def run(self):
        self.valueChanged.emit(0)
        for i in range(100):
            time.sleep(1)  # Do "work"
            self.valueChanged.emit(i)  # Notify progress bar to update via signal
        self.taskFinished.emit()
