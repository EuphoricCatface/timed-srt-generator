import sys
import os
import multiprocessing as mp

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
    QToolTip
)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCursor

import worker


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
        self.hfauth_lineedit.setPlaceholderText("HuggingFace Token: hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        self.hfauth_help_button = QPushButton("?")
        self.hfauth_help_button.clicked.connect(
            lambda: QToolTip.showText(
                QCursor.pos(),
                "Only needed for the initial download.\n"
                "Create your own HuggingFace authentication token and put it here.\n"
                "Make sure to accept the user conditions of `pyannote/segmentation-3.0`\n"
                "and `pyannote/speaker-diarization-3.1`."
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

        # multiprocessing
        self.q_progress = mp.Queue()
        self.q_error_msg = mp.Queue()
        self.q_result = mp.Queue()

        # Keep references to worker process to prevent garbage collection
        self.worker_process = None
        self.worker_check_timer = None

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
        hfauth = self.hfauth_lineedit.text().strip()

        video_path = self.input_lineedit.text().strip()
        if not video_path:
            QMessageBox.warning(self, "Warning", "Please specify a video file.")
            return
        if not os.path.exists(video_path):
            QMessageBox.critical(self, "Error", f"File not found:\n{video_path}")
            return

        srt_path = self.output_lineedit.text().strip()
        if not srt_path:
            QMessageBox.warning(self, "Warning", "Please specify an output path.")
            return
        if os.path.exists(srt_path):
            rtn = QMessageBox.question(self, "Overwrite?", f"{srt_path}\nalready exists. Overwrite?")
            if rtn == QMessageBox.StandardButton.No:
                return

        self.statusBar().showMessage("Running...")

        # Initialize
        while not self.q_progress.empty():
            self.q_progress.get_nowait()
        while not self.q_error_msg.empty():
            self.q_error_msg.get_nowait()
        while not self.q_result.empty():
            self.q_result.get_nowait()

        # Run the worker process
        self.worker_process = mp.Process(
            target=worker.run,
            args=(hfauth, video_path, srt_path, self.q_progress, self.q_error_msg, self.q_result)
        )

        # Connect signals
        def check_worker():
            while not self.q_progress.empty():
                _ = self.q_progress.get_nowait()
                pass  # NYI
            while not self.q_error_msg.empty():
                err_msg = self.q_error_msg.get_nowait()
                self.on_work_error(err_msg)
            if not self.q_result.empty():
                result: bool = self.q_result.get_nowait()
                self.worker_check_timer.stop()
                if result:
                    self.on_diarization_done()
                self.on_work_finished()
                self.worker_process.join()
                self.worker_check_timer = None
                self.worker_process = None

        self.worker_check_timer = QTimer(interval=100)
        self.worker_check_timer.timeout.connect(check_worker)

        # Start the thread
        self.worker_process.start()
        self.worker_check_timer.start()
        self.on_work_started()

    def on_work_started(self):
        """
        Worker signals that it's beginning work (optional).
        """
        # Update UI
        self.hfauth_lineedit.setDisabled(True)
        self.input_lineedit.setDisabled(True)
        self.browse_load_button.setDisabled(True)
        self.output_lineedit.setDisabled(True)
        self.browse_save_button.setDisabled(True)
        self.start_button.setDisabled(True)

    def on_work_error(self, error_msg: str):
        """
        Show an error message from the worker.
        """
        hint_msg = None
        if error_msg.startswith("Pipeline load failed:"):
            hint_msg = ("Create your own HuggingFace token, and enter the token into this program.\n"
                        "Also make sure you have accepted the user conditions of pyannote/segmentation-3.0\n"
                        "and pyannote/speaker-diarization-3.1.")
        elif error_msg == "Failed to extract audio with FFmpeg.":
            hint_msg = "Install FFmpeg"
        if hint_msg:
            display_msg = error_msg + "\n\nHint: " + hint_msg
        else:
            display_msg = error_msg
        QMessageBox.critical(self, "Error", display_msg)
        self.statusBar().showMessage("Ready")

    def on_diarization_done(self):
        """
        Called when diarization completes successfully.
        """
        QMessageBox.information(self, "Success", "Diarization completed successfully!")

    def on_work_finished(self):
        """
        Worker signals that it's finished. Clean up UI or do other tasks.
        """
        self.statusBar().showMessage("Ready")
        self.hfauth_lineedit.setDisabled(False)
        self.input_lineedit.setDisabled(False)
        self.browse_load_button.setDisabled(False)
        self.output_lineedit.setDisabled(False)
        self.browse_save_button.setDisabled(False)
        self.start_button.setDisabled(False)
        # The thread will be quit and deleted, the worker will be deleted.

    def closeEvent(self, event, /):
        # Prevent closing - not the cleanest, but better than the thread left running without the window
        if self.worker_process:
            rtn = QMessageBox.question(self, "Job running", f"Job is running.\n"
                                                            f"Force quit?")
            if rtn == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.worker_process.terminate()
            self.worker_process.join()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()