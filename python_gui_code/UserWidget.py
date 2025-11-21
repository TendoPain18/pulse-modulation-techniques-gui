import math

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QScrollArea, QHBoxLayout, QComboBox, \
    QTableWidget, QTableWidgetItem


class UserWidget(QWidget):
    def __init__(self, Window):
        super().__init__()
        self.Main_Window = Window
        self.screen_width = None
        self.screen_height = None
        self.window_width = None
        self.window_height = None

        self.User_Widget = QWidget()
        self.User_Widget_layout = QVBoxLayout(self.User_Widget)
        self.User_Widget.setStyleSheet("QWidget {background-color: #08A045; border-radius: 10px;}")


        self.User_Input = QWidget()
        self.User_Input_layout = QVBoxLayout(self.User_Input)
        self.User_Input.setStyleSheet("background-color: yellow; border-radius: 10px;")


        self.Choose_Vector_Widget = QWidget()
        self.Choose_Vector = QHBoxLayout(self.Choose_Vector_Widget)
        self.Choose_Vector.setAlignment(Qt.AlignLeft)
        self.Choose_Vector_Label = QLabel("Choose one vector: ")
        self.combo_box = QComboBox()
        self.combo_box.currentIndexChanged.connect(self.on_combo_box_changed)
        self.Choose_Vector.addWidget(self.Choose_Vector_Label)
        self.Choose_Vector.addWidget(self.combo_box)
        self.User_Input_layout.addWidget(self.Choose_Vector_Widget)
        self.User_Widget_layout.addWidget(self.User_Input, 1)



        self.Data_Preview = QWidget()
        self.Data_Preview.setStyleSheet("border-radius: 10px;")
        self.table_widget = QTableWidget(self.Data_Preview)
        self.table_widget.setStyleSheet("background-color: white; border-radius: 10px;")
        self.User_Widget_layout.addWidget(self.Data_Preview, 2)

    def on_combo_box_changed(self, index):
        data = self.Main_Window.data[index]
        rows = len(data)
        columns = len(data[0])
        self.table_widget.setRowCount(rows)
        self.table_widget.setColumnCount(columns)

        for i, row in enumerate(data):
            for j, item in enumerate(row):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(item)))

    def get(self):
        return self.User_Widget

    def set_coordinates(self, sw, sh, ww, wh):
        self.screen_width = sw
        self.screen_height = sh
        self.window_width = ww
        self.window_height = wh

    def ww(self, x):
        return math.floor((self.window_width / 1000) * x)

    def wh(self, x):
        return math.floor((self.window_height / 1000) * x)

    def sw(self, x):
        return math.floor((self.screen_width / 1000) * x)

    def sh(self, x):
        return math.floor((self.screen_height / 1000) * x)

    def set_dimensions(self):
        self.User_Widget_layout.setContentsMargins(10, 10, 10, 10)
        self.table_widget.resize(self.ww(253), self.wh(381))

    def set_stylesheets(self):
        self.Choose_Vector_Label.setStyleSheet(f"font-weight: bold; font-size: {self.sh(10)}px; border-radius: 2px;")
        self.combo_box.setStyleSheet(
            "QComboBox {"
            "   border: 2px solid #555555;"
            "   border-radius: 8px;"
            "   min-width: 6em;"
            "   background: #f5f5f5;"
            "   selection-background-color: darkgray;"
            "}"

            "QComboBox::drop-down {"
            "   border-left: 2px solid #555555;"
            "   border-radius: 0px;"
            "}"
            "QComboBox QAbstractItemView {"
            "   border: 2px solid darkgray;"
            "   selection-background-color: #08A045;"
            "   background: white;"
            "}"
        )

    def set_properties(self):
        self.combo_box.clear()
        self.combo_box.setFixedWidth(self.sw(70))
        self.combo_box.setFixedHeight(self.sw(10))

        for i in self.Main_Window.variables:
            self.combo_box.addItem(i)


