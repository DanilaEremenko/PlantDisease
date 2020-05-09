"""
Custom PyQt5 GUI buttons
"""

from PyQt5.QtWidgets import QPushButton


class ControlButton(QPushButton):
    def __init__(self, text, connect_func, styleSheet=''):
        super(ControlButton, self).__init__()
        btn_width = 64
        btn_height = 32
        self.resize(btn_width, btn_height)
        self.setText(text)
        self.setStyleSheet(styleSheet)
        self.clicked.connect(connect_func)
