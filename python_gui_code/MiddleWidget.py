from PyQt5.QtWidgets import QWidget, QVBoxLayout
from RandomProcess import RandomProcessWindow
from OutputWidget import OutputWidget


class MiddleWidget(QWidget):
    def __init__(self, window):
        super().__init__()
        self.Main_Window = window
        self.mode = 2

        self.Middle_Widget = QWidget()
        self.Middle_Widget_layout = QVBoxLayout(self.Middle_Widget)

        self.RandomProcess = RandomProcessWindow(window)
        self.Output = OutputWidget(window)

    def set_properties(self):
        self.Middle_Widget.setObjectName("Middle_Widget")
        self.Middle_Widget_layout.addWidget(self.RandomProcess.get(), 2)
        self.Middle_Widget_layout.addWidget(self.Output.get(), 1)
        self.RandomProcess.get()
        self.RandomProcess.set_properties()
        self.Output.set_properties()

    def set_stylesheets(self):
        self.Middle_Widget.setStyleSheet("QWidget#Middle_Widget {background-color: #D9D9D9; border-radius: 10px;}")
        self.RandomProcess.set_stylesheets()
        self.Output.set_stylesheets()

    def update_(self):
        self.Middle_Widget.setFixedWidth(self.Main_Window.dimensions.sw(230))
        self.Middle_Widget_layout.setContentsMargins(10, 10, 10, 10)
        self.Output.update_()
        self.RandomProcess.update_()

    def get(self):
        return self.Middle_Widget
