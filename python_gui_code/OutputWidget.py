import math

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QScrollArea, QHBoxLayout


class OutputWidget(QWidget):
    def __init__(self, window):
        super().__init__()
        self.Main_Window = window

        # Create a scroll area
        self.User_Widget = QScrollArea(self)

        self.scroll_contents = QWidget(self.User_Widget)
        self.User_Widget_layout = QVBoxLayout(self.scroll_contents)

        self.Widgets_List = []

    def set_properties(self):
        self.User_Widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.User_Widget.setWidgetResizable(True)
        self.User_Widget.setWidget(self.scroll_contents)
        self.scroll_contents.setObjectName("scroll_contents")
        self.User_Widget_layout.setAlignment(Qt.AlignTop)

    def set_stylesheets(self):
        self.User_Widget.setStyleSheet("QWidget {border-radius: 10px;}")
        self.scroll_contents.setStyleSheet("QWidget#scroll_contents {background-color: #b5b5b5; border: 2px solid black;}")

        self.User_Widget.setStyleSheet("background-color: #b5b5b5; border: 0px solid black; border-radius: 10px;")
        self.User_Widget.verticalScrollBar().setStyleSheet(
            "QScrollBar:vertical { border: 1px solid grey; border-radius: 5px; background: lightgray; width: 8px; margin: 0; }"
            "QScrollBar::handle:vertical {border-radius: 5px; background: #605F5E; min-height: 20px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {border-radius: 5px; height: 0; }")


    def update_(self):
        self.User_Widget.setFixedHeight(self.Main_Window.dimensions.sh(155))
        self.User_Widget_layout.setContentsMargins(10, 10, 10, 10)
        for i in self.Widgets_List:
            i.setStyleSheet(f"font-size: {self.Main_Window.dimensions.sh(13)}px; font-weight: bold;")

    def get(self):
        return self.User_Widget

    def add_text(self, text, boolean=False):
        label = QLabel(text)
        if boolean:
            label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"font-size: {self.Main_Window.dimensions.sh(13)}px;")
        self.Widgets_List.append(label)
        self.User_Widget_layout.addWidget(label)
        self.update_()

    def clear(self):
        while self.User_Widget_layout.count():
            item = self.User_Widget_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.Widgets_List.clear()
