import subprocess
from pathlib import Path

import torch.cuda
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
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return False


def write_diarizaed_to_srt(diarization, output_srt):
    """
    Writes a minimal SRT with speaker labels (optional) and blank lines for text.
    segments: list of (start_time, end_time, speaker_label).
    """

    def srt_time_format(seconds):
        import math
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    # Diarization to segment
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        segments.append((start_time, end_time, speaker))

    # Sort segments by start time to ensure correct SRT order
    segments = sorted(segments, key=lambda x: x[0])

    with open(output_srt, 'w', encoding='utf-8') as f:
        for i, (start, end, speaker) in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{srt_time_format(start)} --> {srt_time_format(end)}\n")
            # Put speaker label as a placeholder comment or prefix
            f.write(f"[{speaker}]: \n\n")


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
        try:
            from pyannote.audio import Pipeline  # Importing this in the global scope creates a lengthy delay at startup
        except Exception as e:
            self.error.emit(f"pyannote import error: {str(e)}")
            self.finished.emit()
            return
        try:
            if Worker.DIARIZATION_PIPELINE is None:
                Worker.DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                     use_auth_token=self.hfauth
                )
            if Worker.DIARIZATION_PIPELINE is None:
                raise RuntimeError("pyannote.audio pipeline not available.")
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
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            Worker.DIARIZATION_PIPELINE.to(device)
            diarization = Worker.DIARIZATION_PIPELINE(audio_output)
        except Exception as e:
            self.error.emit(f"Diarization failed: {str(e)}")
            self.finished.emit()
            return

        # 3) Write SRT output
        try:
            write_diarizaed_to_srt(diarization, self.srt_path)
        except Exception as e:
            self.error.emit(f"SRT output failed: {str(e)}")
            self.finished.emit()
            return

        # Clean up temp file
        try:
            Path(audio_output).unlink(missing_ok=True)
        except OSError:
            pass

        # Emit success
        self.diarization_done.emit()
        self.finished.emit()