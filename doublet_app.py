import sys
import subprocess
import os
import matplotlib
matplotlib.use('Agg')

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QLabel, QMessageBox, QLineEdit,
                             QColorDialog, QStackedWidget, QListWidget, QListWidgetItem)
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
        print("Installing missing dependencies...")
        packages_to_install = [required[pkg] for pkg, ok in zip(required, installed) if not ok]

        for pkg in packages_to_install:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except subprocess.CalledProcessError:
                if pkg in apt_packages and shutil.which("apt"):
                    apt_pkg = apt_packages[pkg]
                    print(f"Failed to install {pkg} via pip. Installing {apt_pkg} via apt...")
                    try:
                        subprocess.check_call(["sudo", "apt", "install", "-y", apt_pkg])
                    except subprocess.CalledProcessError as e:
                        print(f"Error installing {apt_pkg} via apt: {e}")
                        sys.exit(1)
                else:
                    print(f"Failed to install {pkg}. Please check dependencies manually.")
                    sys.exit(1)

install_dependencies()

try:
    from deepvisual import visualize_link_doublet
    from deepcore import sort_duoblet
except ImportError as e:
    print(f"Module import error: {e}")
    print("Please ensure deepvisual.py and deepcore.py are in the same directory")
    sys.exit(1)

class DeepVCApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
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
        self.setWindowTitle('DeepVC')
        self.setGeometry(300, 300, 1000, 700)
        
        icon_path = os.path.join('img', 'logo', 'logo.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"Icon not found at: {icon_path}")

        main_layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        nav_panel = QListWidget()
        nav_panel.setFixedWidth(200)
        nav_panel.addItem(QListWidgetItem("Data Loading"))
        nav_panel.addItem(QListWidgetItem("Data Processing"))
        nav_panel.addItem(QListWidgetItem("Visualization"))
        nav_panel.currentRowChanged.connect(self.display_page)

        self.pages = QStackedWidget()

        # Page 1: File loading
        page_load = QWidget()
        load_layout = QVBoxLayout()
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("font-size: 14px; color: gray;")
        btn_load = QPushButton("Load CSV")
        btn_load.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        btn_load.clicked.connect(self.load_file)
        
        load_layout.addWidget(self.file_label)
        load_layout.addWidget(btn_load)
        load_layout.addStretch()
        page_load.setLayout(load_layout)

        # Page 2: Data processing
        page_process = QWidget()
        process_layout = QVBoxLayout()
        
        self.process_status_label = QLabel("No data loaded")
        self.process_status_label.setStyleSheet("font-size: 14px; color: gray;")
        
        btn_process = QPushButton("Process Data")
        btn_process.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        btn_process.clicked.connect(self.process_data)
        
        btn_save = QPushButton("Save Results")
        btn_save.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
                background-color: #ff9800;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e68a00;
            }
        """)
        btn_save.clicked.connect(self.save_result)

        process_layout.addWidget(self.process_status_label)
        process_layout.addWidget(btn_process)
        process_layout.addWidget(btn_save)
        process_layout.addStretch()
        page_process.setLayout(process_layout)

        # Page 3: Visualization
        page_visual = QWidget()
        visual_layout = QVBoxLayout()

        params_layout = QVBoxLayout()
        self.color_pickers = {}
        color_labels = {
            'loop_color': 'Loop Color',
            'edge_color': 'Edge Color',
            'inter_edge_color': 'Inter Edge Color',
            'background_color': 'Background Color',
            'color_title': 'Title Color'
        }
        
        for param, label_text in color_labels.items():
            btn = QPushButton()
            btn.setFixedSize(80, 30)
            btn.clicked.connect(self.create_color_picker(param))
            self.color_pickers[param] = btn
            label = QLabel(label_text)
            label.setStyleSheet("font-size: 14px;")
            row = QHBoxLayout()
            row.addWidget(label)
            row.addWidget(btn)
            params_layout.addLayout(row)

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Enter visualization title")
        self.title_edit.setStyleSheet("padding: 5px; font-size: 14px;")
        params_layout.addWidget(self.title_edit)

        btn_visualize = QPushButton("Generate Visualization")
        btn_visualize.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
                background-color: #9C27B0;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        btn_visualize.clicked.connect(self.run_visualization)

        visual_layout.addLayout(params_layout)
        visual_layout.addWidget(btn_visualize)
        visual_layout.addStretch()
        page_visual.setLayout(visual_layout)

        self.pages.addWidget(page_load)
        self.pages.addWidget(page_process)
        self.pages.addWidget(page_visual)

        main_layout.addWidget(nav_panel)
        main_layout.addWidget(self.pages)

        self.update_color_buttons()

    def display_page(self, index):
        self.pages.setCurrentIndex(index)
        if index == 1:  # Process page
            if self.current_df is not None:
                self.process_status_label.setText("Data loaded and ready for processing")
                self.process_status_label.setStyleSheet("font-size: 14px; color: green;")
            else:
                self.process_status_label.setText("No data loaded")
                self.process_status_label.setStyleSheet("font-size: 14px; color: red;")

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
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if path:
            try:
                self.current_df = pd.read_csv(path)
                self.file_label.setText(f"Loaded file: {os.path.basename(path)}")
                self.file_label.setStyleSheet("font-size: 14px; color: green;")
                QMessageBox.information(self, "Success", "File loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
                self.file_label.setStyleSheet("font-size: 14px; color: red;")

    def process_data(self):
        if self.current_df is not None:
            try:
                self.current_df = sort_duoblet(self.current_df)
                self.process_status_label.setText("Data processed successfully!")
                self.process_status_label.setStyleSheet("font-size: 14px; color: green;")
                QMessageBox.information(self, "Success", "Data processing completed!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Processing error: {str(e)}")
                self.process_status_label.setText("Processing error")
                self.process_status_label.setStyleSheet("font-size: 14px; color: red;")
        else:
            QMessageBox.warning(self, "Warning", "Please load a file first!")

    def save_result(self):
        if self.current_df is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv)")
            if path:
                if not path.endswith('.csv'):
                    path += '.csv'
                try:
                    self.current_df.to_csv(path, index=False)
                    QMessageBox.information(self, "Success", "File saved successfully!")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "No data to save!")

    def run_visualization(self):
        if self.current_df is not None:
            try:
                save_path, _ = QFileDialog.getSaveFileName(self, "Save Visualization", "", "PNG Files (*.png)")
                if save_path:
                    visualize_link_doublet(
                        self.current_df,
                        loop_color=self.visual_params['loop_color'],
                        edge_color=self.visual_params['edge_color'],
                        inter_edge_color=self.visual_params['inter_edge_color'],
                        background_color=self.visual_params['background_color'],
                        title=self.title_edit.text(),
                        color_title=self.visual_params['color_title']
                    )
                    import matplotlib.pyplot as plt
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close()
                    QMessageBox.information(self, "Success", f"Visualization saved to:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Visualization error: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please load data first!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = DeepVCApp()
    ex.show()
    sys.exit(app.exec())