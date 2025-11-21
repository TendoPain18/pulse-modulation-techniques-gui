from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QScrollArea


class FileWidget(QWidget):
    def __init__(self, window):
        super().__init__()
        self.Main_Window = window

        self.File_Widget = QScrollArea(self)
        self.scroll_contents = QWidget()
        self.File_Widget_layout = QVBoxLayout(self.scroll_contents)

        self.Files = []

    def set_properties(self):
        self.File_Widget.setObjectName("File_Widget")
        self.scroll_contents.setObjectName("scroll_contents")
        self.File_Widget.setWidgetResizable(True)
        self.File_Widget.setWidget(self.scroll_contents)  # ###########################################################
        self.File_Widget_layout.setAlignment(Qt.AlignTop)
        self.add_file("Welcome XD", [], True)

    # ###########################################################################################################
    def set_stylesheets(self):
        self.File_Widget.setStyleSheet("QWidget {background-color: #08A045; border-radius: 10px;}")
        self.scroll_contents.setStyleSheet("QWidget {background-color: #08A045;}")
        self.File_Widget.verticalScrollBar().setStyleSheet(
            "QScrollBar:vertical { border: 1px solid grey; border-radius: 5px; background: lightgray; width: 8px; margin: 0; }"
            "QScrollBar::handle:vertical {border-radius: 5px; background: #605F5E; min-height: 20px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {border-radius: 5px; height: 0; }")

    def update_(self):
        self.File_Widget.setFixedWidth(self.Main_Window.dimensions.sw(130))
        self.File_Widget_layout.setContentsMargins(10, 10, 10, 10)
        self.File_Widget.width()
        for i in self.Files:
            i.layout().itemAt(0).widget().setFixedWidth(self.Main_Window.dimensions.sw(105))
            for j in range(2, i.layout().count()):
                i.layout().itemAt(j).widget().setFixedWidth(self.Main_Window.dimensions.sw(105))
            i.setStyleSheet(f"background-color: #D9D9D9; border-radius: 5px; font-weight: bold; font-size: {self.Main_Window.dimensions.sh(15)}px;")

    def get(self):
        return self.File_Widget

    def add_file(self, file_name, variables, boolean=False):
        file = QWidget()
        file_layout = QVBoxLayout(file)
        label = QLabel(file_name)
        if boolean:
            label.setAlignment(Qt.AlignCenter)
        file_layout.addWidget(label)
        line = QWidget()
        line.setFixedHeight(2)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet("background-color: #333;")
        file_layout.addWidget(line)
        for i in variables:
            var = QLabel(i)
            file_layout.addWidget(var)
        self.Files.append(file)
        self.File_Widget_layout.addWidget(file)
