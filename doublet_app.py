import sys
import subprocess
import os

# Set matplotlib backend to avoid conflicts with PyQt6
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QTabWidget, QLabel, QMessageBox,
                             QLineEdit, QColorDialog, QSpinBox, QFormLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
import pandas as pd

def install_dependencies():
    import shutil

    required = {
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'networkx': 'networkx',
        'PyQt6': 'PyQt6'
    }

    apt_packages = {
        'PyQt6': 'python3-pyqt6'
    }

    installed = []
    for package in required:
        try:
            __import__(package)
            installed.append(True)
        except ImportError:
            installed.append(False)

    if not all(installed):
        print("Установка недостающих зависимостей...")
        packages_to_install = [required[pkg] for pkg, ok in zip(required, installed) if not ok]

        for pkg in packages_to_install:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except subprocess.CalledProcessError:
                if pkg in apt_packages and shutil.which("apt"):
                    apt_pkg = apt_packages[pkg]
                    print(f"Не удалось установить {pkg} через pip. Устанавливаю {apt_pkg} через apt...")
                    try:
                        subprocess.check_call(["sudo", "apt", "install", "-y", apt_pkg])
                    except subprocess.CalledProcessError as e:
                        print(f"Ошибка установки {apt_pkg} через apt: {e}")
                        sys.exit(1)
                else:
                    print(f"Не удалось установить {pkg}. Проверьте зависимости вручную.")
                    sys.exit(1)

# Установка зависимостей перед импортом остальных модулей
install_dependencies()

# Теперь можно импортировать остальные модули
try:
    from deepvisual import visualize_link_doublet
    from deepcore import sort_duoblet
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}")
    print("Проверьте, что файлы deepvisual.py и deepcore.py находятся в той же директории")
    sys.exit(1)

class DoubletApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize visual_params before calling initUI
        self.visual_params = {
            'loop_color': '#FF0000',
            'edge_color': '#000000',
            'inter_edge_color': '#0000FF',
            'background_color': '#FFFFFF',
            'title': '',
            'color_title': '#000000'
        }
        
        self.initUI()
        self.current_df = None

    def initUI(self):
        self.setWindowTitle('Doublet Tools')
        self.setGeometry(300, 300, 800, 600)
        
        # Проверяем наличие иконки
        if os.path.exists('icon.png'):
            self.setWindowIcon(QIcon('icon.png'))

        # Creating tabs
        tabs = QTabWidget()
        self.tab_sort = QWidget()
        self.tab_visual = QWidget()

        tabs.addTab(self.tab_sort, "Сортировка")
        tabs.addTab(self.tab_visual, "Визуализация")

        # Sorting tab
        sort_layout = QVBoxLayout()
        
        self.file_label = QLabel("Файл не выбран")
        btn_load = QPushButton("Загрузить CSV")
        btn_load.clicked.connect(self.load_file)
        
        btn_sort = QPushButton("Сортировать")
        btn_sort.clicked.connect(self.run_sort)
        
        btn_save = QPushButton("Сохранить результат")
        btn_save.clicked.connect(self.save_result)

        sort_layout.addWidget(self.file_label)
        sort_layout.addWidget(btn_load)
        sort_layout.addWidget(btn_sort)
        sort_layout.addWidget(btn_save)
        self.tab_sort.setLayout(sort_layout)

        # Visualization Tab
        visual_layout = QVBoxLayout()
        params_layout = QFormLayout()

        self.color_pickers = {}
        for param in ['loop_color', 'edge_color', 'inter_edge_color', 'background_color', 'color_title']:
            btn = QPushButton()
            btn.setFixedSize(80, 30)
            btn.clicked.connect(self.create_color_picker(param))
            self.color_pickers[param] = btn
            params_layout.addRow(param.replace('_', ' ').title(), btn)

        self.title_edit = QLineEdit()
        params_layout.addRow("Title", self.title_edit)

        btn_visualize = QPushButton("Визуализировать")
        btn_visualize.clicked.connect(self.run_visualization)

        visual_layout.addLayout(params_layout)
        visual_layout.addWidget(btn_visualize)
        self.tab_visual.setLayout(visual_layout)

        self.setCentralWidget(tabs)
        self.update_color_buttons()

    def create_color_picker(self, param):
        def pick_color():
            color = QColorDialog.getColor()
            if color.isValid():
                self.visual_params[param] = color.name()
                self.update_color_buttons()
        return pick_color

    def update_color_buttons(self):
        for param, btn in self.color_pickers.items():
            btn.setStyleSheet(f"background-color: {self.visual_params[param]};")

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                self.current_df = pd.read_csv(path)
                self.file_label.setText(f"Загружен файл: {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки файла: {str(e)}")

    def run_sort(self):
        if self.current_df is not None:
            try:
                self.current_df = sort_duoblet(self.current_df)
                QMessageBox.information(self, "Успех", "Таблица успешно отсортирована!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка сортировки: {str(e)}")
        else:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите файл!")

    def save_result(self):
        if self.current_df is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", "", "CSV Files (*.csv)")
            if path:
                # Убедимся, что файл имеет расширение .csv
                if not path.endswith('.csv'):
                    path += '.csv'
                try:
                    self.current_df.to_csv(path, index=False)  # Сохраняем в формате CSV
                    QMessageBox.information(self, "Успех", "Файл успешно сохранен!")
                except Exception as e:
                 QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения: {str(e)}")
        else:
            QMessageBox.warning(self, "Внимание", "Нет данных для сохранения!")

    def run_visualization(self):
        if self.current_df is not None:
            try:
                # Prompt the user to select a file path for saving the visualization
                save_path, _ = QFileDialog.getSaveFileName(self, "Сохранить визуализацию", "", "PNG Files (*.png)")
                if save_path:
                    # Generate and save the visualization
                    visualize_link_doublet(
                        self.current_df,
                        loop_color=self.visual_params['loop_color'],
                        edge_color=self.visual_params['edge_color'],
                        inter_edge_color=self.visual_params['inter_edge_color'],
                        background_color=self.visual_params['background_color'],
                        title=self.title_edit.text(),
                        color_title=self.visual_params['color_title']
                    )
                    # Save the plot to the specified path
                    import matplotlib.pyplot as plt
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close()
                    
                    # Show a success message with the saved path
                    QMessageBox.information(self, "Успех", f"Визуализация успешно сохранена в файл:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка визуализации: {str(e)}")
        else:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = DoubletApp()
    ex.show()
    sys.exit(app.exec())