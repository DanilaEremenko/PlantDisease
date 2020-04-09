"""
PyQt GUI for main_data_augmentation.py
"""

from PyQt5 import QtWidgets
from pd_gui.components.gui_buttons import ControlButton
from pd_gui.components.gui_layouts import MyGridWidget

import pd_lib.data_maker as dmk
from .gui_window_interface import WindowInterface
from pd_gui.components.gui_labels import ImageTextLabel

import json
import os
import numpy as np


class WindowMultipleExamples(WindowInterface):
    def _init_hbox_control(self):
        self.hbox_control = QtWidgets.QHBoxLayout()
        self.hbox_control.addStretch(1)
        self.hbox_control.addWidget(ControlButton("Okay", self.okay_pressed))
        self.hbox_control.addWidget(ControlButton("Update", self.update_main_layout))
        self.hbox_control.addWidget(ControlButton("Multiple", self.multiple_pressed))
        self.hbox_control.addWidget(ControlButton("Quit", self.quit_default))

    def _define_max_class(self):
        self.max_class = {'name': None, 'num': 0, 'value': None}
        self.max_key_len = 0
        self.max_aug_for_classes = {}
        for key, value in self.classes.items():
            if self.classes[key]['num'] > self.max_class['num']:
                self.max_class['name'] = key
                self.max_class['num'] = self.classes[key]['num']
                self.max_class['value'] = self.classes[key]['value']
            if len(key) > self.max_key_len:
                self.max_key_len = len(key)

            self.max_aug_for_classes[key] = self.classes[key]['num'] \
                                            + int(self.classes[key]['num'] * self.max_aug_part)

    def show_histogram(self, labels, values, title='Diseases distribution'):
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, values, width, label='Examples num')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Num')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)

        fig.tight_layout()

        plt.xticks(rotation=45)
        plt.show()

    def __init__(self, json_list):
        super(WindowMultipleExamples, self).__init__()
        self.postfix = 'joined'

        # TODO maybe will be restored someday
        # with open(self.choose_json(content_title='config gui data')) as gui_config_fp:
        with open('config_gui_diseases.json') as gui_config_fp:
            self.label_size = json.load(gui_config_fp)['qt_label_size']

        # TODO maybe will be restored someday
        # with open(self.choose_json(content_title='config augmentation data')) as aug_config_fp:
        with open('config_augmentation.json') as aug_config_fp:
            aug_config_dict = json.load(aug_config_fp)
            alghs_dict = aug_config_dict['algorithms']
            self.arg_dict = {

                'use_noise': alghs_dict['noise']['use'],
                'intensity_noise_list': alghs_dict['noise']['val_list'],

                'use_deform': alghs_dict['deform']['use'],
                'k_deform_list': alghs_dict['deform']['val_list'],

                'use_blur': alghs_dict['blur']['use'],
                'rad_list': alghs_dict['blur']['val_list'],

                'use_affine': alghs_dict['affine']['use'],
                'affine_list': alghs_dict['affine']['val_list']
            }
            self.max_aug_part = aug_config_dict['max_aug_part']

        if json_list == None:
            json_list = [self.choose_json(content_title='train_data')]
        self.json_name = os.path.splitext(json_list[0])[0]

        if len(json_list) == 1 and self.json_name[-6:] == self.postfix:
            print('Parsing preprocessed json')
            self.classes, img_shape, self.x_data, self.y_data = \
                dmk.json_train_load(json_list[0])
            self.x_data = np.array(self.x_data, dtype='uint8')
            self.y_data = np.array(self.y_data, dtype='uint8')
        else:
            print('Parsing json list')
            self.classes, self.x_data, self.y_data = dmk.get_data_from_json_list(
                json_list=json_list,
                # remove_classes=None,
                # TODO some dev stuff
                remove_classes=['альтернариоз', 'прочие инфекции', 'морщинистая мозаика', 'полосатая мозаика']
            )
        # TODO some dev stuff
        self.classes['здоровые кусты'] = self.classes['марь белая']
        del self.classes['марь белая']

        self._define_max_class()

        self.main_layout = MyGridWidget(hbox_control=self.hbox_control)
        self.setCentralWidget(self.main_layout)

        self.showFullScreen()
        self.update_main_layout()

        print("---------------------------------")
        print('classes      = %s' % str(self.classes))
        print('max_classes  = %s' % str(self.max_aug_for_classes))
        print('ex_num = %d' % sum(map(lambda x: x['num'], self.classes.values())))
        print("---------------------------------")

        self.show_histogram(
            labels=list(self.classes.keys()),
            values=list(map(lambda val: val['num'], self.classes.values()))
        )

    def clear(self):
        self.main_layout.clear()

    def update_main_layout(self):
        self.clear()

        def get_key_by_value(value):
            for key in self.classes.keys():
                if (self.classes[key]['value'] == value).all():
                    return key
            raise Exception('No value == %s' % str(value))

        def add_spaces(word, new_size):  # TODO fix gui label alignment
            while len(word) < new_size:
                word += '_'
            return word

        label_list = []
        for x, y in zip(self.x_data, self.y_data):
            label_list.append(
                ImageTextLabel(
                    x=x,
                    text=add_spaces(get_key_by_value(value=y), new_size=self.max_key_len),
                    label_size=self.label_size
                )
            )
        rect_len = int(np.sqrt(len(self.x_data)))
        self.main_layout.update_grid(
            windows_width=self.main_layout.max_width,
            window_height=self.main_layout.max_height,
            x_len=rect_len,
            y_len=rect_len,
            label_list=label_list
        )

    def okay_pressed(self):
        out_json_path = "%s_%s.json" % (self.json_name, self.postfix)
        print("Save to %s" % out_json_path)

        for key in self.classes.keys():
            self.classes[key]['value'] = list(*self.classes[key]['value'])

        dmk.json_big_create(
            json_path=out_json_path,
            h5_path="%s_%s.hd5f" % (self.json_name, self.postfix),
            x_data=self.x_data,
            y_data=self.y_data,
            longitudes=None,
            latitudes=None,
            img_shape=None,
            classes=self.classes
        )

        self.quit_default()

    def multiple_pressed(self):
        for key, value in self.classes.items():

            if self.classes[key]['num'] < self.max_aug_for_classes[key]:

                max_class_num = self.max_aug_for_classes[key]
                old_class_size = len(self.x_data)
                self.x_data, self.y_data = dmk.multiple_class_examples(x_train=self.x_data, y_train=self.y_data,
                                                                       class_for_multiple=self.classes[key]['value'],
                                                                       **self.arg_dict,
                                                                       max_class_num=max_class_num)

                new_ex_num = len(self.x_data) - old_class_size
                print('%s : generated %d new examples' % (key, new_ex_num))
                self.classes[key]['num'] = 0
                for y in self.y_data:
                    if ((y.__eq__(self.classes[key]['value'])).all()):
                        self.classes[key]['num'] += 1

            else:
                print('%s : generated %d new examples (class_size == max_size)' % (key, 0))
        print("---------------------------------")
        print('classes = %s' % str(self.classes))
        print('ex_num = %d' % sum(map(lambda x: x['num'], self.classes.values())))
        print("---------------------------------")

        self.show_histogram(
            labels=list(self.classes.keys()),
            values=list(map(lambda val: val['num'], self.classes.values())),
            title='Diseases distribution after augmentation'
        )
