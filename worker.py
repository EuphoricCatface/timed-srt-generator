import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, Signal

def extract_audio_with_ffmpeg(video_path: str, output_path: str) -> bool:
    """
    Extract audio from the input video using FFmpeg.
    Returns True if successful, otherwise False.
    """
    cmd = [
        "ffmpeg",
        "-y",           # overwrite output if exists
        "-i", video_path,
        "-vn",          # no video
        "-ac", "1",     # mono
        "-ar", "16000", # 16kHz
        "-f", "wav",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return False

class Worker(QObject):
    """
    Worker to be moved to a background QThread.
    Handles the heavy tasks: FFmpeg extraction + Speaker Diarization.
    """
    # Define signals to communicate back to the main thread
    started = Signal()
    finished = Signal()
    error = Signal(str)
    diarization_done = Signal()  # object will hold the diarization results

    DIARIZATION_PIPELINE = None

    def __init__(self, hfauth, video_path, srt_path, parent=None):
        super().__init__(parent)
        self.hfauth = hfauth
        self.video_path = video_path
        self.srt_path = srt_path
        self._stop_flag = False

    def run(self):
        """
        Main method that is called from the background thread.
        1) Extract audio with FFmpeg
        2) Run pyannote.audio speaker diarization
        """
        self.started.emit()

        # 0) Load the diarization pipeline
        from pyannote.audio import Pipeline  # Importing this in the global scope creates lengthy delay at startup
        if Worker.DIARIZATION_PIPELINE is None:
            try:
                Worker.DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                     use_auth_token=self.hfauth
                )
            except Exception as e:
                self.error.emit(f"Pipeline load failed: {str(e)}")
                self.finished.emit()
                return

        # 1) Extract audio
        audio_output = "temp_audio.wav"
        success = extract_audio_with_ffmpeg(self.video_path, audio_output)
        if not success:
            self.error.emit("Failed to extract audio with FFmpeg.")
            self.finished.emit()
            return

        # 2) Run speaker diarization
        try:
            if not Worker.DIARIZATION_PIPELINE:
                raise RuntimeError("pyannote.audio pipeline not available.")
            diarization = Worker.DIARIZATION_PIPELINE(audio_output)
        except Exception as e:
            self.error.emit(f"Diarization failed: {str(e)}")
            self.finished.emit()
            return

        # 3) Write SRT output
        # ...WIP...

        # Clean up temp file
        try:
            Path(audio_output).unlink(missing_ok=True)
        except OSError:
            pass

        # Emit success
        self.diarization_done.emit(diarization)
        self.finished.emit()