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
from PIL import Image


class WindowMultipleExamples(WindowInterface):
    ##############################################################
    # ---------------- init stuff --------------------------------
    ##############################################################
    def __init__(self, json_list):
        super(WindowMultipleExamples, self).__init__()
        self.postfix = 'joined'

        self.label_size = (224, 224)

        # TODO maybe will be restored someday
        # with open(self.choose_json(content_title='config augmentation data')) as aug_config_fp:
        with open('config_data_augmentation.json') as aug_config_fp:
            aug_config_dict = json.load(aug_config_fp)
            alghs_dict = aug_config_dict['algorithms']

            self.max_aug_part = aug_config_dict['aug_part']
            self.augment_all = aug_config_dict['augment_all']
            self.output_json = aug_config_dict['output_json']
            self.save_data_binary = aug_config_dict['save_data_to_binary']
            self.save_data_dir = aug_config_dict['save_data_to_dir']

            for key, value in alghs_dict.items():
                if value['use'] and len(value['val_list']) != self.max_aug_part:
                    raise Exception('bad val_list size for %s' % key)

            self.arg_dict = {

                'use_noise': alghs_dict['noise']['use'],
                'intensity_noise_list': alghs_dict['noise']['val_list'],

                'use_deform': alghs_dict['deform']['use'],
                'k_deform_list': alghs_dict['deform']['val_list'],

                'use_blur': alghs_dict['blur']['use'],
                'rad_list': alghs_dict['blur']['val_list'],

                'use_affine': alghs_dict['affine']['use'],
                'affine_list': alghs_dict['affine']['val_list'],

                'use_contrast': alghs_dict['contrast']['use'],
                'contrast_list': alghs_dict['contrast']['val_list']
            }

        if json_list == None:
            raise Exception('No passed json')
        self.json_name = os.path.splitext(json_list[0])[0]

        if len(json_list) == 1:
            print('Parsing preprocessed json')
            self.classes, img_shape, self.x_data, self.y_data = \
                dmk.json_big_load(json_list[0])
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
        self.init_size = len(self.x_data)
        self.img_shape = self.x_data.shape[1:]
        # TODO some dev stuff
        # self.classes['здоровый куст'] = self.classes['марь белая']
        # del self.classes['марь белая']
        # self.classes['мозаика'] = self.classes['прочие мозаики']
        # del self.classes['прочие мозаики']
        # self.classes['сорняк'] = self.classes['прочие сорняки']
        # del self.classes['прочие сорняки']

        self._define_max_class()

        self.main_layout = MyGridWidget(hbox_control=self.hbox_control)
        self.setCentralWidget(self.main_layout)

        self.showFullScreen()
        self.show_full()

        print("---------------------------------")
        print('classes      = %s' % str(self.classes))
        print('max_classes  = %s' % str(self.max_aug_for_classes))
        print('ex_num = %d' % sum(map(lambda x: x['num'], self.classes.values())))
        print("---------------------------------")

        self.show_histogram(
            labels=list(self.classes.keys()),
            values=list(map(lambda val: val['num'], self.classes.values()))
        )

    def _init_hbox_control(self):
        self.hbox_control = QtWidgets.QHBoxLayout()
        self.hbox_control.addStretch(1)
        self.hbox_control.addWidget(ControlButton("Okay", self.okay_pressed))
        self.hbox_control.addWidget(ControlButton("Show Full", self.show_full))
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

    ##############################################################
    # ---------------- gui logic stuff ---------------------------
    ##############################################################
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

    def clear(self):
        self.main_layout.clear()

    def show_full(self):
        self.clear()

        def get_key_by_value(value):
            for key in self.classes.keys():
                if (self.classes[key]['value'] == value).all():
                    return key
            raise Exception('No value == %s' % str(value))

        label_list = []
        for x, y in zip(self.x_data, self.y_data):
            label_text = get_key_by_value(value=y)
            label_text += (self.max_key_len - len(label_text)) * " "
            label_list.append(
                ImageTextLabel(
                    x=x,
                    text=label_text,
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
        print("Save to %s" % self.output_json)

        for key in self.classes.keys():
            self.classes[key]['value'] = list(self.classes[key]['value'])

        if self.save_data_binary:
            ###################################################################################
            # ------------------------ be aware of SIGKILL ------------------------------------
            ###################################################################################
            binary_path = dir_path = "/".join(self.output_json.split('/')[:-1]) + "/" \
                                     + self.output_json.split('/')[-1][:-5] + ".hd5f"
            dmk.json_big_create(
                json_path=self.output_json,
                h5_path=binary_path,
                x_data=self.x_data,
                y_data=self.y_data,
                longitudes=None,
                latitudes=None,
                img_shape=None,
                classes=self.classes
            )
        elif self.save_data_dir:
            ###################################################################################
            # we can not pass array with size 10_000 * 256 * 256 * 3 to function, shout SIGKILL
            ###################################################################################
            dir_path = "/".join(self.output_json.split('/')[:-1]) \
                       + "/" + self.output_json.split('/')[-1][:-5]

            id = []
            label = []

            if not dir_path:
                raise Exception("Data directory %s need to be existed" % dir_path)

            for key in self.classes.keys():
                self.classes[key]['value_num'] = dmk.get_num_from_pos(self.classes[key]['value'])
                self.classes[key]['saved_num'] = 0
            for i, (x, y) in enumerate(zip(self.x_data, self.y_data)):
                for key in self.classes.keys():
                    if (self.classes[key]['value'] == y).all():
                        file_path = "%s/%d.JPG" % (dir_path, i + 1)
                        Image.fromarray(x).save(file_path)
                        print("%s saved" % file_path)
                        id.append(file_path)
                        label.append(key)

            with open(self.output_json, "w") as fp:
                json.dump(
                    {
                        "classes": self.classes, "img_shape": None,
                        "dir_path": dir_path,
                        "longitudes": None, "latitudes": None,
                        "dataframe": {
                            "id": id,
                            "label": label
                        }
                    },
                    fp)
                fp.close()

        self.quit_default()

    def multiple_pressed(self):

        if self.augment_all:
            if sum(map(lambda x: x['num'], self.classes.values())) < self.init_size * self.max_aug_part:
                # ----------------------------------- augment all -----------------------------------------------------
                old_class_size = len(self.x_data)
                x_data_new, y_data_new = dmk.multiple_class_examples(x_train=self.x_data[:self.init_size],
                                                                     y_train=self.y_data[:self.init_size],
                                                                     **self.arg_dict,
                                                                     max_class_num=self.init_size * self.max_aug_part,
                                                                     mode='all')

                self.x_data = np.append(self.x_data, x_data_new)
                self.y_data = np.append(self.y_data, y_data_new)

                ex_num = int(self.y_data.shape[0] / len(self.classes))

                self.x_data.shape = (ex_num, *self.img_shape)
                self.y_data.shape = (ex_num, len(self.classes))

                for key, value in self.classes.items():
                    new_ex_num = len(self.x_data) - old_class_size
                    print('%s : generated %d new examples' % (key, new_ex_num))
                    self.classes[key]['num'] = 0
                    for y in self.y_data:
                        if ((y.__eq__(self.classes[key]['value'])).all()):
                            self.classes[key]['num'] += 1
            else:
                print('ex_num = max_num')


        else:
            # ----------------------------------- augment by classes ---------------------------------------------
            for key, value in self.classes.items():

                if self.classes[key]['num'] < self.max_aug_for_classes[key]:

                    max_class_num = self.max_aug_for_classes[key]
                    old_class_size = len(self.x_data)

                    x_data_new, y_data_new = dmk.multiple_class_examples(x_train=self.x_data[:self.init_size],
                                                                         y_train=self.y_data[:self.init_size],
                                                                         class_for_mult=self.classes[key]['value'],
                                                                         **self.arg_dict,
                                                                         max_class_num=max_class_num)

                    self.x_data = np.append(self.x_data, x_data_new)
                    self.y_data = np.append(self.y_data, y_data_new)

                    ex_num = int(self.y_data.shape[0] / len(self.classes))

                    self.x_data.shape = (ex_num, *self.img_shape)
                    self.y_data.shape = (ex_num, len(self.classes))

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

        self.show_augmentation()

    def show_augmentation(self):
        self.clear()

        label_list = []

        for i in [0, 1, 2]:
            label_list.append(
                ImageTextLabel(
                    x=self.x_data[i],
                    text='original',
                    label_size=self.label_size
                )
            )
            for j in range(1, int(self.max_aug_part) + 1):
                label_text = 'aug %.2f' % self.arg_dict['contrast_list'][j - 1]
                label_list.append(
                    ImageTextLabel(
                        x=self.x_data[i + j * self.init_size],
                        text=label_text,
                        label_size=self.label_size
                    ),
                )

        x_len = j + 1
        y_len = int(len(label_list) / x_len)
        self.main_layout.update_grid(
            windows_width=self.main_layout.max_width,
            window_height=self.main_layout.max_height,
            x_len=x_len,
            y_len=y_len,
            label_list=label_list
        )
