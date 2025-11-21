import random
import math
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QProgressBar
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap


class StartWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.index = 0
        self.screen_width = None
        self.screen_height = None
        self.window_width = None
        self.window_height = None
        self.x_coordinate = None
        self.y_coordinate = None
        self.text_list = ["Initialization...", "Loading Resources...", "Checking for Updates...", "Authentication...",
                          "Finishing..."]

        # Create a label and pixmap
        self.label = QLabel(self)
        self.pixmap = QPixmap("images/bg - Copy.jpg")
        self.Top_Label = QLabel("R2023a Update 5 (9.14.0.2337262)\n64-bit (win64)\nJuly 24, 2023", self)
        self.Start_Label = QLabel("Starting now", self)
        self.Description_Label = QLabel(self.text_list[0], self)
        self.Bottom_Label = QLabel("C 1984-2023 The MathWorks, Inc. Protected by U.S and international patents. See math-works.com/patents. MATLAB\nand Simulink are registered trademarks of The MathWorks, Inc. See math-works.com/trademarks for a list of additional\ntrademarks. Other product or brand names may be trademarks or registered trademarks of their respective holders.", self)
        self.progress_bar = QProgressBar(self)
        self.timer = QTimer(self)

        self.get_screen_dimensions()
        self.set_properties()
        self.set_stylesheets()
        self.set_dimensions()

    def ww(self, x):
        return math.floor((self.window_width / 1000) * x)

    def wh(self, x):
        return math.floor((self.window_height / 1000) * x)

    def sw(self, x):
        return math.floor((self.screen_width / 1000) * x)

    def sh(self, x):
        return math.floor((self.screen_height / 1000) * x)

    def set_dimensions(self):
        self.label.setGeometry(0, 0, self.window_width, self.window_height)
        scaled = self.pixmap.scaled(self.window_width, self.window_height)
        self.label.setPixmap(scaled)
        self.Top_Label.setGeometry(self.ww(35), self.wh(30), self.window_width, self.wh(120))
        self.Start_Label.setGeometry(self.ww(35), self.wh(420), self.window_width, 60)
        self.Description_Label.setGeometry(self.ww(35), self.wh(475), self.window_width, 60)
        self.Bottom_Label.setGeometry(self.ww(110), self.wh(850), self.window_width, self.wh(130))
        self.progress_bar.setGeometry(0, self.window_height - (self.window_height // 100), self.window_width, self.window_height // 100)

    def set_stylesheets(self):
        self.Top_Label.setStyleSheet(
            f"color: rgba(255, 183, 6, 0.7); font-size: {self.wh(24)}px; font-weight: bold;"
        )
        self.Start_Label.setStyleSheet(
            f"color: rgba(255, 183, 6, 0.8); font-size: {self.wh(55)}px; font-weight: bold;"
        )
        self.Description_Label.setStyleSheet(
            f"color: rgba(255, 183, 6, 0.6); font-size: {self.wh(25)}px; font-weight: bold;"
        )
        self.progress_bar.setStyleSheet(
            "QProgressBar {background-color: red;}"
            "QProgressBar::chunk {background-color: #D09217;}"
        )
        self.Bottom_Label.setStyleSheet(
            f"color: rgba(255, 183, 6, 0.6); font-size: {self.wh(24)}px; font-weight: bold;"
        )

    def set_properties(self):
        self.setGeometry(self.x_coordinate, self.y_coordinate, self.window_width, self.window_height)
        self.setMinimumSize(800, 450)
        self.setWindowFlags(Qt.FramelessWindowHint)
        # self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.label.setPixmap(self.pixmap)

        self.progress_bar.setTextVisible(False)

        self.timer.timeout.connect(self.update_progress_bar)
        self.timer.start(1000)

    def get_screen_dimensions(self):
        # Get screen dimensions
        screen_geometry = QApplication.desktop().screenGeometry()
        self.screen_width = screen_geometry.width()
        self.screen_height = screen_geometry.height()

        # Calculate window position and size
        self.window_width = int(self.screen_width * 0.35)
        self.window_height = int(self.screen_height * 0.37)
        self.x_coordinate = (self.screen_width - self.window_width) // 2
        self.y_coordinate = (self.screen_height - self.window_height) // 2

    def update_progress_bar(self):
        current_value = self.progress_bar.value()
        if current_value < 100:
            # Increment the progress bar value with a random step
            step = random.randint(20, 25)
            if current_value + step > 100:
                current_value = 100 - step
            self.progress_bar.setValue(current_value + step)
            self.Description_Label.setText(self.text_list[self.index])
            if self.index < len(self.text_list) - 1:
                self.index += 1
        else:
            self.timer.stop()
            self.close()

    def get_window_dimensions(self):
        new_size = self.size()
        self.window_width = new_size.width()
        self.window_height = new_size.height()

    def resizeEvent(self, event):
        self.get_window_dimensions()
        self.set_dimensions()
        self.set_stylesheets()
