from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QHBoxLayout, QComboBox, \
    QTableWidget, QTableWidgetItem, QLineEdit


class RandomProcessWindow(QWidget):
    def __init__(self, window):
        super().__init__()
        self.Main_Window = window

        self.index_1 = -1
        self.index_2 = -1
        self.index_3 = 0

        self.User_Widget = QWidget()
        self.User_Widget_layout = QVBoxLayout(self.User_Widget)

        self.User_Input = QWidget()
        self.User_Input_layout = QVBoxLayout(self.User_Input)

        self.Choose_Vector_Widget = QWidget()
        self.Choose_Vector = QHBoxLayout(self.Choose_Vector_Widget)
        self.Choose_Vector_Label = QLabel("Time vector: ")
        self.combo_box = QComboBox()

        self.Choose_Vector_Widget_2 = QWidget()
        self.Choose_Vector_2 = QHBoxLayout(self.Choose_Vector_Widget_2)
        self.Choose_Vector_Label_2 = QLabel("Sample variable: ")
        self.combo_box_2 = QComboBox()

        self.Choose_Vector_Widget_3 = QWidget()
        self.Choose_Vector_3 = QHBoxLayout(self.Choose_Vector_Widget_3)
        self.Choose_Vector_Label_3 = QLabel("Modulation Type: ")
        self.combo_box_3 = QComboBox()

        self.M_Vector_Widget_2 = QWidget()
        self.M_Vector_2 = QHBoxLayout(self.M_Vector_Widget_2)
        self.M_label = QLabel('  Enter Ts:')
        self.M_line_edit = QLineEdit()

        self.nth_Vector_Widget_2 = QWidget()
        self.nth_Vector_2 = QHBoxLayout(self.nth_Vector_Widget_2)
        self.nth_label = QLabel('   Enter T:')
        self.nth_line_edit = QLineEdit()

        self.A_Vector_Widget = QWidget()
        self.A_Vector = QHBoxLayout(self.A_Vector_Widget)
        self.A_label = QLabel('   Enter F:')
        self.A_line_edit = QLineEdit()

        self.Data_Preview = QWidget()
        self.Data_Preview_layout = QVBoxLayout(self.Data_Preview)
        self.table_widget = QTableWidget(self.Data_Preview)

        self.Data_Preview_2 = QWidget()
        self.Data_Preview_layout_2 = QVBoxLayout(self.Data_Preview_2)
        self.table_widget_2 = QTableWidget(self.Data_Preview_2)

        self.update_combo()

    def set_properties(self):
        self.User_Input_layout.setSpacing(0)
        self.User_Widget.setObjectName("User_Widget")
        self.User_Input.setObjectName("User_Input")
        self.Data_Preview.setObjectName("Data_Preview")
        self.Data_Preview_2.setObjectName("Data_Preview_2")
        self.table_widget.setObjectName("table_widget")
        self.table_widget_2.setObjectName("table_widget_2")
        self.Choose_Vector_Label.setObjectName("Choose_Vector_Label")
        self.Choose_Vector_Label_2.setObjectName("Choose_Vector_Label_2")
        self.Choose_Vector_Label_3.setObjectName("Choose_Vector_Label_3")
        self.M_Vector_Widget_2.setObjectName("M_Vector_Widget_2")
        self.nth_Vector_Widget_2.setObjectName("nth_Vector_Widget_2")
        self.A_Vector_Widget.setObjectName("A_Vector_Widget")
        self.User_Input_layout.setAlignment(Qt.AlignTop)
        self.User_Input_layout.setContentsMargins(0, 0, 0, 0)
        self.User_Input_layout.addWidget(self.Choose_Vector_Widget)
        self.User_Input_layout.addWidget(self.Choose_Vector_Widget_2)
        self.User_Input_layout.addWidget(self.Choose_Vector_Widget_3)

        self.Choose_Vector.setAlignment(Qt.AlignLeft)
        self.combo_box.currentIndexChanged.connect(self.on_combo_box_changed)
        self.Choose_Vector.addWidget(self.Choose_Vector_Label)
        self.Choose_Vector.addWidget(self.combo_box)
        self.Choose_Vector.addWidget(self.M_label)  # ####################################################
        self.Choose_Vector.addWidget(self.M_line_edit)  # ################################################

        self.Choose_Vector_2.setAlignment(Qt.AlignLeft)
        self.combo_box_2.currentIndexChanged.connect(self.on_combo_box_changed_2)
        self.Choose_Vector_2.addWidget(self.Choose_Vector_Label_2)
        self.Choose_Vector_2.addWidget(self.combo_box_2)
        self.Choose_Vector_2.addWidget(self.nth_label)  # ####################################################
        self.Choose_Vector_2.addWidget(self.nth_line_edit)  # ################################################

        self.Choose_Vector_3.setAlignment(Qt.AlignLeft)
        self.combo_box_3.currentIndexChanged.connect(self.on_combo_box_changed_3)
        self.Choose_Vector_3.addWidget(self.Choose_Vector_Label_3)
        self.Choose_Vector_3.addWidget(self.combo_box_3)
        self.Choose_Vector_3.addWidget(self.A_label)  # ####################################################
        self.Choose_Vector_3.addWidget(self.A_line_edit)  # ################################################

        self.M_Vector_2.setContentsMargins(0, 0, 0, 0)
        self.M_Vector_2.setAlignment(Qt.AlignLeft)
        # self.M_Vector_2.addWidget(self.M_label)
        # self.M_Vector_2.addWidget(self.M_line_edit)
        self.User_Input_layout.addWidget(self.M_Vector_Widget_2)

        self.nth_Vector_2.setContentsMargins(0, 0, 0, 0)
        self.nth_Vector_2.setAlignment(Qt.AlignLeft)
        # self.nth_Vector_2.addWidget(self.nth_label)
        # self.nth_Vector_2.addWidget(self.nth_line_edit)
        self.User_Input_layout.addWidget(self.nth_Vector_Widget_2)

        self.A_Vector.setContentsMargins(0, 0, 0, 0)
        self.A_Vector.setAlignment(Qt.AlignLeft)
        # self.nth_Vector_2.addWidget(self.nth_label)
        # self.nth_Vector_2.addWidget(self.nth_line_edit)
        self.User_Input_layout.addWidget(self.A_Vector_Widget)

        self.Data_Preview_layout.addWidget(self.table_widget)
        self.Data_Preview_layout_2.addWidget(self.table_widget_2)

        self.User_Widget_layout.addWidget(self.User_Input, 10)
        self.User_Widget_layout.addWidget(self.Data_Preview, 1)
        self.User_Widget_layout.addWidget(self.Data_Preview_2, 1)

        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_widget_2.setEditTriggers(QTableWidget.NoEditTriggers)

    def set_stylesheets(self):
        self.User_Widget.setStyleSheet("QWidget#User_Widget {border-radius: 10px;}")
        self.User_Input.setStyleSheet("QWidget#User_Input {background-color: #C8B88A; border-radius: 10px;}")
        self.Data_Preview.setStyleSheet("QWidget#Data_Preview {border-radius: 10px; background-color: white;}")
        self.Data_Preview_2.setStyleSheet("QWidget#Data_Preview_2 {border-radius: 10px; background-color: white;}")
        self.table_widget.setStyleSheet("QTableWidget#table_widget {background-color: white; border-radius: 10px;}")
        self.table_widget_2.setStyleSheet("QTableWidget#table_widget_2 {background-color: white; border-radius: 10px;}")
        self.combo_box.setStyleSheet("QComboBox {border-radius: 15px;")
        self.combo_box_2.setStyleSheet("QComboBox {border-radius: 15px;")
        self.combo_box_3.setStyleSheet("QComboBox {border-radius: 15px;")

    def update_(self):
        self.User_Widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.User_Widget_layout.setContentsMargins(10, 10, 10, 10)
        self.User_Input.setFixedHeight(self.Main_Window.dimensions.sh(145))
        self.combo_box.setFixedWidth(self.Main_Window.dimensions.sw(64))
        self.combo_box.setFixedHeight(self.Main_Window.dimensions.sw(12))
        self.combo_box_2.setFixedWidth(self.Main_Window.dimensions.sw(49))
        self.combo_box_2.setFixedHeight(self.Main_Window.dimensions.sw(12))
        self.combo_box_3.setFixedWidth(self.Main_Window.dimensions.sw(49))
        self.combo_box_3.setFixedHeight(self.Main_Window.dimensions.sw(12))
        self.M_line_edit.setStyleSheet(f"border-radius: 5px; font-weight: bold; font-size: {self.Main_Window.dimensions.sh(13)}px")
        self.nth_line_edit.setStyleSheet(f"border-radius: 5px; font-weight: bold; font-size: {self.Main_Window.dimensions.sh(13)}px")
        self.A_line_edit.setStyleSheet(f"border-radius: 5px; font-weight: bold; font-size: {self.Main_Window.dimensions.sh(13)}px")
        self.M_line_edit.setFixedWidth(self.Main_Window.dimensions.sw(35))
        self.M_line_edit.setFixedHeight(self.Main_Window.dimensions.sw(12))
        self.nth_line_edit.setFixedWidth(self.Main_Window.dimensions.sw(35))
        self.nth_line_edit.setFixedHeight(self.Main_Window.dimensions.sw(12))
        self.A_line_edit.setFixedWidth(self.Main_Window.dimensions.sw(35))
        self.A_line_edit.setFixedHeight(self.Main_Window.dimensions.sw(12))
        self.Choose_Vector_Label.setFixedWidth(self.Main_Window.dimensions.sw(45))
        self.Choose_Vector_Label_2.setFixedWidth(self.Main_Window.dimensions.sw(60))
        self.Choose_Vector_Label_3.setFixedWidth(self.Main_Window.dimensions.sw(60))
        self.M_label.setFixedWidth(self.Main_Window.dimensions.sw(37.5))
        self.nth_label.setFixedWidth(self.Main_Window.dimensions.sw(38))
        self.A_label.setFixedWidth(self.Main_Window.dimensions.sw(38))
        self.Choose_Vector.setContentsMargins(self.Main_Window.dimensions.sw(7), self.Main_Window.dimensions.sw(9),
                                              self.Main_Window.dimensions.sw(5), self.Main_Window.dimensions.sw(4))
        self.Choose_Vector_2.setContentsMargins(self.Main_Window.dimensions.sw(7), self.Main_Window.dimensions.sw(6),
                                                self.Main_Window.dimensions.sw(5), self.Main_Window.dimensions.sw(9))
        self.Choose_Vector_3.setContentsMargins(self.Main_Window.dimensions.sw(7), self.Main_Window.dimensions.sw(2),
                                                self.Main_Window.dimensions.sw(5), self.Main_Window.dimensions.sw(6))

        self.Choose_Vector_Label.setStyleSheet(
            f"QLabel#Choose_Vector_Label {{font-weight: bold; font-size: {self.Main_Window.dimensions.sh(13)}px;}}")
        self.Choose_Vector_Label_2.setStyleSheet(
            f"QLabel#Choose_Vector_Label_2 {{font-weight: bold; font-size: {self.Main_Window.dimensions.sh(13)}px;}}")
        self.Choose_Vector_Label_3.setStyleSheet(
            f"QLabel#Choose_Vector_Label_3 {{font-weight: bold; font-size: {self.Main_Window.dimensions.sh(13)}px;}}")
        self.M_label.setStyleSheet(
            f"QLabel {{font-weight: bold; font-size: {self.Main_Window.dimensions.sh(13)}px;}}")
        self.nth_label.setStyleSheet(
            f"QLabel {{font-weight: bold; font-size: {self.Main_Window.dimensions.sh(13)}px;}}")
        self.A_label.setStyleSheet(
            f"QLabel {{font-weight: bold; font-size: {self.Main_Window.dimensions.sh(13)}px;}}")

    def get(self):
        return self.User_Widget

    def on_combo_box_changed(self, index):
        self.index_1 = index
        data = self.Main_Window.data[index]
        rows = len(data)
        columns = len(data[0])
        self.table_widget.setRowCount(rows)
        self.table_widget.setColumnCount(columns)

        for i, row in enumerate(data):
            for j, item in enumerate(row):
                item_widget = QTableWidgetItem(str(item))
                item_widget.setSizeHint(QSize(10, 10))  # Set the size of the item
                self.table_widget.setItem(i, j, item_widget)

    def on_combo_box_changed_2(self, index):
        self.index_2 = index
        data = self.Main_Window.data[index]
        rows = len(data)
        columns = len(data[0])
        self.table_widget_2.setRowCount(rows)
        self.table_widget_2.setColumnCount(columns)

        for i, row in enumerate(data):
            for j, item in enumerate(row):
                self.table_widget_2.setItem(i, j, QTableWidgetItem(str(item)))

    def on_combo_box_changed_3(self, index):
        self.index_3 = index
        if index == 0:
            self.M_label.setText('  Enter Ts:')
            self.nth_label.setText('   Enter T:')
            self.A_label.setText('   Enter F:')
        elif index == 1:
            self.M_label.setText('  Enter Ts:')
            self.nth_label.setText('   Enter T:')
            self.A_label.setText('   Enter F:')
        elif index == 2:
            self.M_label.setText('  Enter Ts:')
            self.nth_label.setText('   Enter A:')
            self.A_label.setText('   Enter F:')
        elif index == 3:
            self.M_label.setText('  Enter Ts:')
            self.nth_label.setText('   Enter T:')
            self.A_label.setText('  Enter A,F:')


    def update_combo(self):
        self.combo_box.clear()
        for i in self.Main_Window.variables:
            self.combo_box.addItem(i)

        self.combo_box_2.clear()
        for i in self.Main_Window.variables:
            self.combo_box_2.addItem(i)

        self.combo_box_3.clear()
        names = ["PAM", "Flat-Top PAM", "PWM", "PPM"]
        for i in range(4):
            self.combo_box_3.addItem(names[i])
