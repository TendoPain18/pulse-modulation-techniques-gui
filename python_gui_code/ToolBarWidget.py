from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QFrame, QSpacerItem
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from tkinter import filedialog
import scipy.io
import numpy as np
from pathlib import Path


class ToolBarWidget(QWidget):
    def __init__(self, window):
        super().__init__()
        self.Main_Window = window

        self.ToolBar_Widget = QWidget()
        self.ToolBar_layout = QHBoxLayout(self.ToolBar_Widget)

        self.buttons_names = ["Start", "Stop", "Load", "Save"]
        self.buttons_images = ["images/start.png", "images/stop.png", "images/load.png",
                               "images/save.png"]
        self.button_functions = []
        self.buttons = []
        self.spaces = []

    def set_properties(self):
        self.ToolBar_Widget.setObjectName("ToolBar_Widget")
        self.add_functions_to_list()
        self.create_buttons()

    def set_stylesheets(self):
        self.ToolBar_Widget.setStyleSheet("QWidget#ToolBar_Widget {background-color: #D9D9D9;}")

    def update_(self):
        # ToolBar_layout
        self.ToolBar_layout.setContentsMargins(self.Main_Window.dimensions.sw(2), self.Main_Window.dimensions.sw(2),
                                               self.Main_Window.dimensions.sw(4), self.Main_Window.dimensions.sw(2))
        self.ToolBar_layout.setSpacing(0)

        # buttons
        for i in self.buttons:
            i.setIconSize(QSize(self.Main_Window.dimensions.sw(20), self.Main_Window.dimensions.sw(20)))
            i.setFixedSize(self.Main_Window.dimensions.sw(55), self.Main_Window.dimensions.sw(23))

        for i in self.buttons:
            i.setStyleSheet("QPushButton {color: black; font-weight: bold; border-radius: 10px}"
                            f"QPushButton {{font-size: {self.Main_Window.dimensions.sw(8)}px;}}"
                            "QPushButton:hover { border: 1px solid black; background-color: #EEF5FF}")

        # spaces
        for i in range(len(self.spaces) - 1):
            self.spaces[i].changeSize(self.Main_Window.dimensions.sw(5), 0)
        self.spaces[-1].changeSize(self.Main_Window.dimensions.sw(1500), 0)

    def get(self):
        return self.ToolBar_Widget

    def create_buttons(self):
        for i in range(len(self.buttons_names)):
            button = QPushButton(self.buttons_names[i])
            button.setIcon(QIcon(self.buttons_images[i]))
            button.setFlat(True)
            button.clicked.connect(self.button_functions[i])
            self.add_spacer()
            self.ToolBar_layout.addWidget(button)
            self.add_spacer()
            self.add_toolbar_line()
            self.buttons.append(button)
        self.add_spacer()

    def add_spacer(self):
        spacer = QSpacerItem(0, 0)
        self.spaces.append(spacer)
        self.ToolBar_layout.addItem(spacer)

    def add_toolbar_line(self):
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        self.ToolBar_layout.addWidget(line)

    def start_function(self):
        self.Main_Window.start_random_process()

    @staticmethod
    def stop_function():
        print("Stop button clicked")

    def load_function(self):
        file_path = filedialog.askopenfilename(title="Choose a .mat file", filetypes=[("MAT files", "*.mat")])
        file_name = Path(file_path).name
        if file_path == "":
            return
        # Load the .mat file
        mat_data = scipy.io.loadmat(file_path)

        # Create a list to store variables
        variables_list = []

        # Iterate through the variables in the loaded data
        for var_name, var_data in mat_data.items():
            # Skip metadata fields
            if var_name.startswith("__"):
                continue

            # Check if the variable is a 2D array (list of lists)
            if isinstance(var_data, np.ndarray) and var_data.ndim == 2:
                # Convert the NumPy array to a list of lists
                var_data_list = var_data.tolist()
                # Append the variable name and its data to the list
                variables_list.append((var_name, var_data_list))
            elif isinstance(var_data, np.ndarray) and var_data.ndim == 1:
                # If it's a 1D array, convert it to a list
                var_data_list = var_data.tolist()
                # Append the variable name and its data to the list
                variables_list.append((var_name, var_data_list))
            else:
                # For other data types, handle as needed
                variables_list.append((var_name, var_data))
        self.Main_Window.add_to_file_widgets(file_name, variables_list)

    @staticmethod
    def save_function():
        print("Save button clicked")

    def random_process_function(self):
        self.Main_Window.Middle.set_mode(2)

    def add_functions_to_list(self):
        self.button_functions.append(self.start_function)
        self.button_functions.append(self.stop_function)
        self.button_functions.append(self.load_function)
        self.button_functions.append(self.save_function)

