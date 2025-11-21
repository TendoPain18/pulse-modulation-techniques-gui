from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
from ToolBarWidget import ToolBarWidget
from FileWidget import FileWidget
from MiddleWidget import MiddleWidget
from PlotsWidget import PlotsWidget
from Calculations import Calculations
from Dimensions import Dimensions
from PyQt5.QtWidgets import QMessageBox
import time


class ProgramWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # initialize variables
        self.dimensions = Dimensions()
        self.calc = Calculations()
        self.file_names = []
        self.variables = []
        self.data = []
        self.vector_index_1 = -1
        self.vector_index_2 = -1
        self.vector_index_3 = 0
        self.m = ""
        self.nth = ""
        self.A = ""

        # Create widgets
        self.Main_Widget = QWidget(self)
        self.ToolBar = ToolBarWidget(self)
        self.Bottom_Window_Widget = QWidget()
        self.File = FileWidget(self)
        self.Middle = MiddleWidget(self)
        self.Plots = PlotsWidget(self)

        # create layouts
        self.Main_Widget_layout = QVBoxLayout(self.Main_Widget)
        self.Bottom_Window_layout = QHBoxLayout(self.Bottom_Window_Widget)

        # start
        self.start()

    def start(self):
        self.dimensions.initialize(self)
        self.set_properties()
        self.set_stylesheets()
        self.update_()

    def set_properties(self):
        # window
        self.setGeometry(self.dimensions.x_coordinate, self.dimensions.y_coordinate,
                         self.dimensions.window_width, self.dimensions.window_height)
        self.setWindowTitle("Communication Theory Project")
        self.setWindowIcon(QIcon("images/bg - Copy - Copy - Copy-modified.png"))
        # main widget layout
        self.Main_Widget.setObjectName("Main_Widget")
        self.Main_Widget_layout.setContentsMargins(0, 0, 0, 0)
        self.Main_Widget_layout.setSpacing(0)
        self.Main_Widget_layout.addWidget(self.ToolBar.get(), 1)
        self.Main_Widget_layout.addWidget(self.Bottom_Window_Widget, 15)

        # bottom part of window layout
        self.Bottom_Window_Widget.setObjectName("Bottom_Window_Widget")
        self.Bottom_Window_layout.addWidget(self.File.get())
        self.Bottom_Window_layout.addWidget(self.Middle.get())
        self.Bottom_Window_layout.addWidget(self.Plots.get())

        # widgets
        self.ToolBar.set_properties()
        self.File.set_properties()
        self.Middle.set_properties()
        self.Plots.set_properties()

    def set_stylesheets(self):
        self.Main_Widget.setStyleSheet("QWidget#Main_Widget {background-color: #0F2167;}")
        self.Bottom_Window_Widget.setStyleSheet("QWidget#Bottom_Window_Widget {background-color: #0B6E4F;}")
        self.ToolBar.set_stylesheets()
        self.File.set_stylesheets()
        self.Middle.set_stylesheets()
        self.Plots.set_stylesheets()

    def update_(self):
        self.setMinimumSize(self.dimensions.sw(600), self.dimensions.sh(600))
        self.Main_Widget.setGeometry(0, 0, self.dimensions.window_width, self.dimensions.window_height)
        self.ToolBar.update_()
        self.File.update_()
        self.Middle.update_()
        self.Plots.update_()

    def resizeEvent(self, event):
        self.dimensions.update(self)
        self.update_()

    def start_random_process(self):
        self.vector_index_1 = self.Middle.RandomProcess.index_1
        self.vector_index_2 = self.Middle.RandomProcess.index_2
        self.vector_index_3 = self.Middle.RandomProcess.index_3
        self.m = self.Middle.RandomProcess.M_line_edit.text()
        self.nth = self.Middle.RandomProcess.nth_line_edit.text()
        self.A = self.Middle.RandomProcess.A_line_edit.text()

        if self.vector_index_3 == 0:
            result = self.calc.natural_pam_modulation(self.data[self.vector_index_1][0],
                                                      self.data[self.vector_index_2][0],
                                                      float(self.m), float(self.nth), float(self.A))
            self.Middle.Output.clear()
            self.Plots.update_chart(1, result[1], result[0], result[2], [])
            self.Plots.update_chart(2, result[3], result[0], result[4], [])
            self.Plots.update_chart(3, result[5], result[0], result[6], [])
            self.Plots.update_chart(4, result[7], result[0], result[8], [])

        elif self.vector_index_3 == 1:
            result = self.calc.flat_top_pam_modulation(self.data[self.vector_index_1][0],
                                                       self.data[self.vector_index_2][0],
                                                       float(self.m), float(self.nth), float(self.A))
            self.Middle.Output.clear()
            self.Plots.update_chart(1, result[1], result[0], result[2], [])
            self.Plots.update_chart(2, result[3], result[0], result[4], [])
            self.Plots.update_chart(3, result[5], result[0], result[6], [])
            self.Plots.update_chart(4, result[7], result[0], result[8], [])

        elif self.vector_index_3 == 2:
            result = self.calc.pwm_modulation(self.data[self.vector_index_1][0],
                                              self.data[self.vector_index_2][0],
                                              float(self.m), float(self.nth), float(self.A))
            self.Middle.Output.clear()
            self.Plots.update_chart(1, result[1], result[0], result[2], [])
            self.Plots.update_chart(2, result[3], result[0], result[4], [])
            self.Plots.update_chart(3, result[5], result[0], result[6], [])
            self.Plots.update_chart(4, result[7], result[0], result[8], [])

        elif self.vector_index_3 == 3:
            parts = self.A.split(",")
            # Convert the substrings to integers
            f1 = float(parts[0])
            f2 = float(parts[1])
            result = self.calc.ppm_modulation(self.data[self.vector_index_1][0],
                                              self.data[self.vector_index_2][0],
                                              float(self.m), float(self.nth), f1, f2)

            self.Middle.Output.clear()
            self.Plots.update_chart(1, result[1], result[0], result[2], [])
            self.Plots.update_chart(2, result[3], result[0], result[4], [])
            self.Plots.update_chart(3, result[5], result[0], result[6], [])
            self.Plots.update_chart(4, result[7], result[0], result[8], [])

        # state = self.error_handle(2)
        # if state:
        #     return


    def add_to_file_widgets(self, filename, data):
        self.file_names.append(filename)

        variables = [entry[0] for entry in data]
        variable_data = [entry[1] for entry in data]

        self.variables.extend(variables)
        self.data.extend(variable_data)

        self.File.add_file(filename, variables)
        self.File.update_()
        self.Middle.RandomProcess.update_combo()

    def error_handle(self, mode):
        if mode == 2:
            if self.vector_index_1 == -1:
                self.show_error_message("Time Vector", "Please choose time vector")
                return True
            if self.vector_index_2 == -1:
                self.show_error_message("Message", "Please choose message signal")
                return True
            if self.m == "":
                self.show_error_message("Ts Value", "Please enter Ts value")
                return True
            if self.nth == "":
                self.show_error_message("M value", "Please enter N value")
                return True
            if not (isinstance(self.data[self.vector_index_1][0], list) and all(
                    isinstance(item, (int, float)) for item in self.data[self.vector_index_1][0]) and len(
                self.data[self.vector_index_1]) == 1):
                self.show_error_message("Time Vector", "Time vector should be one list of numbers only")
                return True
            if not all(
                    isinstance(sublist, list) and all(isinstance(num, (int, float)) for num in sublist) for sublist in
                    self.data[self.vector_index_2]):
                self.show_error_message("Random Process", "Random process should be list of list of numbers only")
                return True
            if not self.m.isnumeric():
                self.show_error_message("M Value", "M value should be numeric")
                return True
            if not self.nth.isnumeric():
                self.show_error_message("N Value", "N value should be numeric")
                return True
            if int(self.nth) < 1 or int(self.nth) > len(self.data[self.vector_index_2]):
                self.show_error_message("N Value", "Enter a Valid Index of N")
                return True
            t_len = len(self.data[self.vector_index_1][0])
            for i in self.data[self.vector_index_2]:
                if len(i) > t_len:
                    self.show_error_message("Logic Error", "All sample functions length should equal to time length")
                    return True

    @staticmethod
    def show_error_message(title, message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setText(title)
        error_dialog.setInformativeText(message)
        error_dialog.setWindowTitle("Error")
        error_dialog.setFixedSize(800, 200)
        error_dialog.setMinimumWidth(800)
        error_dialog.exec_()
