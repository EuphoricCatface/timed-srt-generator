import sys
import os

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QLabel,
    QMessageBox,
    QToolTip
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QCursor


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blank Subtitle Generator with PyAnnote")
        self.resize(500, 150)

        # Central widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Layouts
        main_layout = QVBoxLayout(main_widget)
        hfauth_layout = QHBoxLayout()
        input_layout = QHBoxLayout()
        output_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        # Widgets
        self.hfauth_lineedit = QLineEdit()
        self.hfauth_lineedit.setPlaceholderText("hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        self.hfauth_help_button = QPushButton("?")
        self.hfauth_help_button.clicked.connect(
            lambda: QToolTip.showText(
                QCursor.pos(),
                "Create your own HuggingFace authentication token and put it here."
            )
        )

        self.input_lineedit = QLineEdit()
        self.input_lineedit.setPlaceholderText("Input video file path...")

        self.browse_load_button = QPushButton("Browse")
        self.browse_load_button.clicked.connect(self.browse_file_load)

        self.output_lineedit = QLineEdit()
        self.output_lineedit.setPlaceholderText("Output .srt file path...")

        self.browse_save_button = QPushButton("Browse")
        self.browse_save_button.clicked.connect(self.browse_file_save)

        self.browse_load_button = QPushButton("Browse")
        self.browse_load_button.clicked.connect(self.browse_file_load)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_processing)

        # Add widgets to layouts
        hfauth_layout.addWidget(self.hfauth_lineedit)
        hfauth_layout.addWidget(self.hfauth_help_button)

        input_layout.addWidget(self.input_lineedit)
        input_layout.addWidget(self.browse_load_button)

        output_layout.addWidget(self.output_lineedit)
        output_layout.addWidget(self.browse_save_button)

        button_layout.addWidget(self.start_button)

        main_layout.addLayout(hfauth_layout)
        main_layout.addLayout(input_layout)
        main_layout.addLayout(output_layout)
        main_layout.addLayout(button_layout)

        # Status bar
        self.statusBar().showMessage("Ready")

        # Keep references to thread & worker to prevent garbage collection
        self.thread = None
        self.worker = None

    def browse_file_load(self):
        """
        Opens a file dialog to select a video file,
        sets the QLineEdit text to the selected path.
        """
        file_dialog = QFileDialog(self, "Select Video File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if not file_dialog.exec():
            return
        selected_files = file_dialog.selectedFiles()
        if not selected_files:
            return
        self.input_lineedit.setText(selected_files[0])
        # Autocomplete output path
        video_path = selected_files[0]
        path_split = os.path.split(video_path)
        ext_split = os.path.splitext(path_split[1])
        srt_path = os.path.join(path_split[0], ext_split[0] + ".srt")
        self.output_lineedit.setText(srt_path)

    def browse_file_save(self):
        """
        Opens a file dialog to select a video file,
        sets the QLineEdit text to the selected path.
        """
        file_dialog = QFileDialog(self, "Select Video File")
        file_dialog.setFileMode(QFileDialog.AnyFile)
        if not file_dialog.exec():
            return
        selected_files = file_dialog.selectedFiles()
        if not selected_files:
            return
        self.output_lineedit.setText(selected_files[0])

    def start_processing(self):
        """
        Spawns a QThread for the heavy tasks.
        """
        pass


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()