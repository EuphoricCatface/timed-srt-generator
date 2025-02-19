import subprocess
from pathlib import Path
import os.path
import enum
import multiprocessing as mp

import torch.cuda


class WorkerProgress(enum.Enum):
    Initializing = enum.auto()
    PipelineLoading = enum.auto()
    AudioExtracting = enum.auto()
    Diarizing = enum.auto()
    SRTOutput = enum.auto()
    Finalizing = enum.auto()


DIARIZATION_PIPELINE = None


def extract_audio_with_ffmpeg(video_path: str, output_path: str) -> bool:
    """
    Extract audio from the input video using FFmpeg.
    Returns True if successful, otherwise False.
    """
    if os.name == "nt":
        # If on Windows, find ffmpeg.exe in the script folder
        ffmpeg = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ffmpeg.exe")
    else:
        ffmpeg = "ffmpeg"
    cmd = [
        ffmpeg,
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


def run(hfauth, video_path, srt_path, q_progress: mp.Queue, q_error_msg: mp.Queue, q_result: mp.Queue):
    """
    Main method that is called from the background thread.
    0) Load the diarization pipeline
    1) Extract audio with FFmpeg
    2) Run pyannote.audio speaker diarization
    3) Write an SRT output file
    """

    global DIARIZATION_PIPELINE
    q_progress.put(WorkerProgress.Initializing)

    # 0) Load the diarization pipeline
    q_progress.put(WorkerProgress.PipelineLoading)
    try:
        from pyannote.audio import Pipeline  # Importing this in the global scope creates a lengthy delay at startup
    except Exception as e:
        q_error_msg.put(f"pyannote import error: {str(e)}")
        q_result.put(False)
        return
    try:
        if DIARIZATION_PIPELINE is None:
            DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                 use_auth_token=hfauth
            )
        if DIARIZATION_PIPELINE is None:
            raise RuntimeError("pyannote.audio pipeline not available.")
    except Exception as e:
        q_error_msg.put(f"Pipeline load failed: {str(e)}")
        q_result.put(False)
        return

    # 1) Extract audio
    q_progress.put(WorkerProgress.AudioExtracting)
    audio_filename = "temp_audio.wav"
    audio_output_folder = os.path.dirname(srt_path)
    audio_output = os.path.join(audio_output_folder, audio_filename)
    success = extract_audio_with_ffmpeg(video_path, audio_output)
    if not success:
        q_error_msg.put("Failed to extract audio with FFmpeg.")
        q_result.put(False)
        return

    # 2) Run speaker diarization
    q_progress.put(WorkerProgress.Diarizing)
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        DIARIZATION_PIPELINE.to(device)
        diarization = DIARIZATION_PIPELINE(audio_output)
    except Exception as e:
        q_error_msg.put(f"Diarization failed: {str(e)}")
        q_result.put(False)
        return

    # 3) Write an SRT output file
    q_progress.put(WorkerProgress.SRTOutput)
    try:
        write_diarizaed_to_srt(diarization, srt_path)
    except Exception as e:
        q_error_msg.put(f"SRT output failed: {str(e)}")
        q_result.put(False)
        return

    # Clean up temp file
    q_progress.put(WorkerProgress.Finalizing)
    try:
        Path(audio_output).unlink(missing_ok=True)
    except OSError:
        pass

    # Success
    q_result.put(True)
    return