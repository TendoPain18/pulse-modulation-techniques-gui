import math

from PyQt5.QtWidgets import QApplication


class Dimensions:
    def __init__(self):
        self.screen_width = None
        self.screen_height = None
        self.window_width = None
        self.window_height = None
        self.x_coordinate = None
        self.y_coordinate = None

    def initialize(self, window):
        self.get_screen_dimensions(window)
        self.window_width = int(self.screen_width * 0.65)
        self.window_height = int(self.screen_height * 0.72)
        self.x_coordinate = (self.screen_width - self.window_width) // 2
        self.y_coordinate = (self.screen_height - self.window_height) // 2

    def get_window_dimensions(self, window):
        new_size = window.size()
        self.window_width = new_size.width()
        self.window_height = new_size.height()

    def get_screen_dimensions(self, window):
        screen_number = QApplication.desktop().screenNumber(window)
        screen_geometry = QApplication.desktop().screenGeometry(screen_number)
        self.screen_width = screen_geometry.width()
        self.screen_height = screen_geometry.height()
        # self.screen_width = 1920
        # self.screen_height = 1080

    def update(self, window):
        self.get_screen_dimensions(window)
        self.get_window_dimensions(window)

    def ww(self, x):
        return math.floor((self.window_width / 1000) * x)

    def wh(self, x):
        return math.floor((self.window_height / 1000) * x)

    def sw(self, x):
        temp = math.floor((self.screen_width / 1000) * x)
        return temp

    def sh(self, x):
        temp = math.floor((self.screen_height / 1000) * x)
        return temp
