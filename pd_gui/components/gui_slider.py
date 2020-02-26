"""
Custom PyQt5 GUI slider
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider
import numpy as np


class ImgSizeSlider(QSlider):
    def __init__(self, min_val=1024, max_val=4096, step_num=6, orientation='vertical'):
        if orientation == 'vertical':
            super(ImgSizeSlider, self).__init__(Qt.Vertical)
        elif orientation == 'horizontal':
            super(ImgSizeSlider, self).__init__(Qt.Horizontal)
        else:
            raise Exception("orientation can be vertical or horizontal")
        step = int((max_val - min_val) / step_num)

        self.val_list = np.linspace(min_val, max_val, step_num)
        self.setMinimum(min_val)
        self.setMaximum(max_val)
        self.setTickPosition(QSlider.TicksBelow)
        self.setTickInterval(step)
        self.valueChanged.connect(self.value_controller)

    def value_controller(self, value):
        for i in range(0, len(self.val_list) - 1):
            if self.val_list[i] <= value < self.val_list[i + 1]:
                self.setValue(self.val_list[i])
                # print("Slider Value = %d" % self.value())
                return
            elif self.val_list[i] < value <= self.val_list[i + 1]:
                self.setValue(self.val_list[i + 1])
                # print("Slider Value = %d" % self.value())
                return
