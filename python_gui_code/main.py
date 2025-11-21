from StartWindow import StartWindow
from ProgramWindow import ProgramWindow
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exec_()

    random_process_window = ProgramWindow()
    random_process_window.show()
    app.exec_()

