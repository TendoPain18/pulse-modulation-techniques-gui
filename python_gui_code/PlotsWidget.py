import random
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotsWidget(QWidget):
    def __init__(self, window):
        super().__init__()
        self.Main_Window = window

        # Initialize data lists
        self.plot_1_x = []
        self.plot_1_y = []
        self.plot_1_z = []
        self.plot_2_x = []
        self.plot_2_y = []
        self.plot_2_z = []
        self.plot_3_x = []
        self.plot_3_y = []
        self.plot_3_z = []
        self.plot_4_x = []
        self.plot_4_y = []
        self.plot_4_z = []

        self.Plots_Widget = QWidget()
        self.Plots_Widget_layout = QVBoxLayout(self.Plots_Widget)
        self.Top_Horizontal_Layout = QHBoxLayout()
        self.Bottom_Horizontal_Layout = QHBoxLayout()

        self.Chart_layouts = []
        self.Charts = []
        self.Chart_Widgets = []

    def set_properties(self):
        self.Plots_Widget.setObjectName("Plots_Widget")
        for i in range(1, 5):
            self.Charts.append(Figure())
            chart_x = getattr(self, f"plot_{i}_x")
            chart_y = getattr(self, f"plot_{i}_y")
            chart_z = getattr(self, f"plot_{i}_z")
            self.update_chart(i, [f"chart {i}", "x-axis", "y-axis"], chart_x, chart_y, chart_z)
            wid = QWidget()
            chart_layout = QVBoxLayout(wid)
            chart_layout.addWidget(FigureCanvas(self.Charts[i - 1]))
            chart_layout.setObjectName("chart_layout")
            self.Chart_layouts.append(chart_layout)
            self.Chart_Widgets.append(wid)
            if i < 3:
                self.Top_Horizontal_Layout.addWidget(wid)
            else:
                self.Bottom_Horizontal_Layout.addWidget(wid)
        self.Plots_Widget_layout.addLayout(self.Top_Horizontal_Layout)
        self.Plots_Widget_layout.addLayout(self.Bottom_Horizontal_Layout)

    def set_stylesheets(self):
        self.Plots_Widget.setStyleSheet("QWidget#Plots_Widget {background-color: #08A045; border-radius: 10px;}")
        for i in self.Chart_Widgets:
            i.setStyleSheet("QWidget {border-radius: 10px; background-color: white; margin: 5px;}")

    def update_(self):
        self.Plots_Widget_layout.setContentsMargins(15, 15, 15, 15)

    def get(self):
        return self.Plots_Widget

    def update_chart(self, num, names, x_values, y_values, z_values):
        setattr(self, f'plot_{num}_x', x_values)
        setattr(self, f'plot_{num}_y', y_values)
        setattr(self, f'plot_{num}_z', z_values)
        self.Charts[num-1].clf()
        if len(z_values) == 0 and len(x_values) > 0 and len(y_values) > 0:
            ax = self.Charts[num - 1].add_subplot(111)
            if isinstance(y_values, list) and all(isinstance(sublist, list) for sublist in y_values):
                for i in y_values:
                    if i[0] == "x":
                        ax.axvline(x=i[1], ymin=i[2], ymax=i[3], color='r', linewidth=1)
                    elif i[0] == "y":
                        ax.axvline(y=i[1], xmin=i[2], xmax=i[3], color='r', linewidth=1)
                    elif i[0] == "xlim":
                        ax.set_xlim(i[1], i[2])
                    elif i[0] == "ylim":
                        ax.set_ylim(i[1], i[2])
                    else:
                        ax.plot(x_values, i)
            else:
                ax.plot(x_values, y_values)
            ax.set_title(names[0])
            ax.set_xlabel(names[1])
            ax.set_ylabel(names[2])
            if num == 1:
                self.Charts[num - 1].savefig(f'plots/message_signal.png')
            elif num == 2:
                self.Charts[num - 1].savefig(f'plots/pulse_signal.png')
            elif num == 3:
                self.Charts[num - 1].savefig(f'plots/modulated_signal.png')
            elif num == 4:
                self.Charts[num - 1].savefig(f'plots/demodulated_signal.png')
        elif len(z_values) > 0 and len(x_values) > 0 and len(y_values) > 0:
            ax = self.Charts[num - 1].add_subplot(111, projection='3d')
            surf = ax.plot_surface(x_values, y_values, z_values, cmap='viridis')  # Use plot_surface instead of scatter
            ax.set_title(names[0])
            ax.set_xlabel(names[1])
            ax.set_ylabel(names[2])
            ax.set_zlabel(names[3])
        else:
            ax = self.Charts[num - 1].add_subplot(111)
            ax.plot([], [])
            ax.set_title(names[0])
            ax.set_xlabel(names[1])
            ax.set_ylabel(names[2])
        self.Charts[num-1].canvas.draw()
